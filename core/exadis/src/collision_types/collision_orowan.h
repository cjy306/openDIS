/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    CollisionOrowan
 *    Implements Orowan bypass for spherical precipitates and
 *    twin-boundary blocking for planar obstacles.
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_COLLISION_OROWAN_H
#define EXADIS_COLLISION_OROWAN_H

// collision_retroactive.h is already included by collision.h before this file
// Do not re-include it here to avoid circular dependency issues.

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        CollisionOrowan
 *
 *-------------------------------------------------------------------------*/
class CollisionOrowan : public CollisionRetroactive {
public:

    CollisionOrowan(System* system) : CollisionRetroactive(system) {}

    /*-----------------------------------------------------------------------
     *  handle_orowan  (Kokkos parallel)
     *---------------------------------------------------------------------*/
    void handle_orowan(System* system)
    {
        if (system->obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nnodes = net->Nnodes_local;
        int Nobs = (int)system->obstacles.size();

        Kokkos::View<SphericalObstacle*, T_memory_space> d_obs("d_obs", Nobs);
        auto h_obs = Kokkos::create_mirror_view(d_obs);
        for (int j = 0; j < Nobs; j++) h_obs(j) = system->obstacles[j];
        Kokkos::deep_copy(d_obs, h_obs);

        auto nodes = net->get_nodes();
        auto segs  = net->get_segs();
        auto conn  = net->get_conn();

        Kokkos::parallel_for("OrowanNodes", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos = nodes[i].pos;

            for (int j = 0; j < Nobs; j++) {
                // Breakaway point obstacles are cut through, never bypassed:
                // no sphere geometry applies (else a just-depinned node sitting
                // at the center would be teleported by R to the shell).
                if (d_obs(j).type == OBSTACLE_BREAKAWAY) continue;

                Vec3   dv    = pos - d_obs(j).center;
                double dist2 = dv.norm2();
                double R     = d_obs(j).radius;
                double r2    = R * R;

                // Thin shell around the sphere: also catch nodes slightly outside
                // the surface so near-surface loop nodes are projected consistently.
                // This removes the in/out flip-flop (one node snapped, its neighbor
                // left free) that produced the jagged loop. The projection target is
                // still the radius-R intersection circle (uses r2 = R*R below);
                // the shell only widens the catch zone.
                double Rs    = 1.025 * R;  // shell outer radius (~10b @ R=391b); tune via the 0.025 factor
                if (dist2 < Rs * Rs) {
                    // Find glide plane normal from connected segments
                    Vec3 n_glide(0.0);
                    int nconn = conn[i].num;
                    for (int k = 0; k < nconn; k++) {
                        Vec3 sp = segs[conn[i].seg[k]].plane;
                        double sp_norm = sp.norm();
                        if (sp_norm > 1e-10) {
                            n_glide = (1.0 / sp_norm) * sp;
                            break;
                        }
                    }

                    if (n_glide.norm2() > 0.5) {
                        // Project sphere center onto glide plane
                        Vec3 C = d_obs(j).center;
                        double h = dot(C - pos, n_glide);
                        Vec3 C_proj = C - h * n_glide;

                        // Intersection circle radius
                        double r_circle2 = r2 - h * h;
                        if (r_circle2 > 0.0) {
                            double r_circle = sqrt(r_circle2);
                            // Direction from C_proj to node (in glide plane)
                            Vec3 d_dir = pos - C_proj;
                            double d_norm = d_dir.norm();
                            if (d_norm > 1e-10) {
                                d_dir = (1.0 / d_norm) * d_dir;
                            } else {
                                // Node is at center projection, pick arbitrary in-plane direction
                                Vec3 arb = (fabs(n_glide.x) < 0.9) ? Vec3(1,0,0) : Vec3(0,1,0);
                                d_dir = cross(n_glide, arb).normalized();
                            }
                            // Project to nearest point on intersection circle
                            nodes[i].pos = C_proj + r_circle * d_dir;
                        } else {
                            // Glide plane doesn't intersect sphere, fallback to radial
                            double dist = sqrt(dist2);
                            Vec3 normal = (dist > 1e-10) ? (1.0/dist) * dv : Vec3(0,0,1);
                            nodes[i].pos = C + R * normal;
                        }
                    } else {
                        // No valid glide plane, fallback to radial projection
                        double dist = sqrt(dist2);
                        Vec3 normal = (dist > 1e-10) ? (1.0/dist) * dv : Vec3(0,0,1);
                        nodes[i].pos = d_obs(j).center + R * normal;
                    }
                    return;
                }
            }
        });
        Kokkos::fence();
    }

    /*-----------------------------------------------------------------------
     *  handle_twin_wall  (Kokkos parallel over nodes)
     *
     *  Region-based twin-plane enforcement.  For each planar obstacle,
     *  the "active side" is the side that contains the box center.
     *  Any node found on the opposite side is projected back onto the
     *  plane.  No xold needed — purely based on current position.
     *---------------------------------------------------------------------*/
    void handle_twin_wall(System* system)
    {
        if (system->planar_obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nnodes  = net->Nnodes_local;
        int Nplanes = (int)system->planar_obstacles.size();

        Kokkos::View<PlanarObstacle*, T_memory_space> d_planes("d_planes_wall", Nplanes);
        auto h_planes = Kokkos::create_mirror_view(d_planes);
        for (int j = 0; j < Nplanes; j++) h_planes(j) = system->planar_obstacles[j];
        Kokkos::deep_copy(d_planes, h_planes);

        auto nodes = net->get_nodes();
        auto segs  = net->get_segs();
        auto conn  = net->get_conn();
        auto cell  = net->cell;
        Vec3 box_center = cell.center();

        // Load spherical obstacles for proximity check
        int Nobs = (int)system->obstacles.size();
        Kokkos::View<SphericalObstacle*, T_memory_space> d_obs_tw("d_obs_tw", Nobs > 0 ? Nobs : 1);
        if (Nobs > 0) {
            auto h_obs_tw = Kokkos::create_mirror_view(d_obs_tw);
            for (int k = 0; k < Nobs; k++) h_obs_tw(k) = system->obstacles[k];
            Kokkos::deep_copy(d_obs_tw, h_obs_tw);
        }

        Kokkos::View<int, T_memory_space> d_count("d_count_wall");
        Kokkos::deep_copy(d_count, 0);

        Kokkos::parallel_for("TwinWall", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos = nodes[i].pos;

            // Skip nodes near precipitates (Orowan mechanism takes priority)
            for (int k = 0; k < Nobs; k++) {
                double dist = (pos - d_obs_tw(k).center).norm();
                if (dist < d_obs_tw(k).radius * 2.0) return;
            }

            for (int j = 0; j < Nplanes; j++) {
                Vec3   n_twin = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d = dot(pos - point, n_twin);
                double center_d = dot(box_center - point, n_twin);

                // Node is on the opposite side of center → project back
                if (d * center_d < 0.0) {
                    // Collect unique glide plane normals from connected segments
                    int nconn = conn[i].num;
                    Vec3 n1(0.0), n2(0.0);
                    int nplanes_found = 0;
                    for (int k = 0; k < nconn; k++) {
                        Vec3 sp = segs[conn[i].seg[k]].plane;
                        double sp_norm = sp.norm();
                        if (sp_norm < 1e-10) continue;
                        sp = (1.0 / sp_norm) * sp;
                        if (nplanes_found == 0) {
                            n1 = sp;
                            nplanes_found = 1;
                        } else if (nplanes_found == 1) {
                            // Check if different from n1
                            double cr = cross(n1, sp).norm();
                            if (cr > 1e-4) {
                                n2 = sp;
                                nplanes_found = 2;
                                break;
                            }
                        }
                    }

                    if (nplanes_found == 1) {
                        // Single glide plane: project along glide-plane direction
                        // proj_dir = n_twin - dot(n_twin, n_glide) * n_glide
                        Vec3 proj_dir = n_twin - dot(n_twin, n1) * n1;
                        double pdn = dot(proj_dir, n_twin);
                        if (fabs(pdn) > 1e-10) {
                            double t = -d / pdn;
                            nodes[i].pos = pos + t * proj_dir;
                        } else {
                            // Glide plane parallel to twin plane, skip
                            continue;
                        }
                    } else if (nplanes_found == 2) {
                        // Junction: project along junction line
                        Vec3 L = cross(n1, n2);
                        double L_norm = L.norm();
                        if (L_norm > 1e-10) {
                            L = (1.0 / L_norm) * L;
                            double ldn = dot(L, n_twin);
                            if (fabs(ldn) > 0.1) {
                                double t = -d / ldn;
                                nodes[i].pos = pos + t * L;
                            } else {
                                // Junction line nearly parallel to twin plane, fallback
                                nodes[i].pos = pos - d * n_twin;
                            }
                        } else {
                            // Degenerate, fallback to vertical projection
                            nodes[i].pos = pos - d * n_twin;
                        }
                    } else {
                        // No valid glide plane found, fallback to vertical projection
                        nodes[i].pos = pos - d * n_twin;
                    }

                    Kokkos::atomic_inc(&d_count());
                    return;
                }
            }
        });
        Kokkos::fence();

        auto h_count = Kokkos::create_mirror_view(d_count);
        Kokkos::deep_copy(h_count, d_count);
        if (h_count() > 0)
            printf("[TwinWall] projected %d node(s)\n", h_count());
    }

    /*-----------------------------------------------------------------------
     *  point_in_tri  (planar point-in-triangle test in the glide plane)
     *      a,b,c : triangle vertices (assumed coplanar with normal n)
     *      p     : query point;  returns true if p is inside triangle abc
     *---------------------------------------------------------------------*/
    static bool point_in_tri(const Vec3& a, const Vec3& b, const Vec3& c,
                             const Vec3& p, const Vec3& n)
    {
        double d1 = dot(cross(b - a, p - a), n);
        double d2 = dot(cross(c - b, p - b), n);
        double d3 = dot(cross(a - c, p - c), n);
        bool has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        bool has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
        return !(has_neg && has_pos);
    }

    /*-----------------------------------------------------------------------
     *  handle_breakaway  (serial network)
     *
     *  Foreman-Makin point-obstacle (cutting) mechanism, complementary to
     *  the Orowan bypass in handle_orowan().  When a dislocation segment
     *  sweeps across an obstacle during this step, a node is inserted at
     *  the obstacle and pinned (v=0 via PINNED_NODE).  The pinned node is
     *  released when the resultant of its unit arm tangents exceeds
     *  2*cos(phi_crit/2) -- the N-arm generalization of the breakaway-angle
     *  criterion (identical to the segment-angle test for a 2-arm node).
     *
     *  Runs on the serial network because split_seg() is a topological
     *  operation; placed right after CollisionRetroactive::handle() so the
     *  serial copy is fresh, before handle_orowan() syncs to the device.
     *---------------------------------------------------------------------*/
    void handle_breakaway(System* system)
    {
        if (system->obstacles.empty()) return;

        SerialDisNet* network = system->get_serial_network();
        double dt   = system->realdt;

        // 步初位置真值(积分器每步存 system->xold;访问范式同 retroactive_collision)。
        // 用记录替代 pos - dt*v 反推:不再依赖"步内匀速直线"假设,积分后被挪过的
        // 节点(如 retroactive 碰撞处理)其位移自动纳入扫掠四边形。
#if EXADIS_FULL_UNIFIED_MEMORY
        T_x& xold = system->xold;
#else
        T_x::HostMirror xold = Kokkos::create_mirror_view(system->xold);
        Kokkos::deep_copy(xold, system->xold);
#endif
        int nxold = (int)xold.extent(0);   // 当步新生节点(索引越界)退回速度反推

        int    Nobs = (int)system->obstacles.size();
        int    ncap = 0, nrel = 0, npin = 0;   // diagnostics

        // ---- Phase A: capture (swept area over point obstacle) ----
        // 本步段扫过的四边形 [o1,o2,p2,p1](o = pos - dt*v 为步初位置)若包含障碍在滑移面内的
        // 投影 C_proj,即认为线扫过该障碍 -> 在 C_proj 插一个钉节点(回退到障碍处)。
        // 扫掠能抓"一步跨过"的快线;垂直门 h^2<R^2 限定障碍要贴近该段滑移面。
        // 仅端点护栏防同段相邻臂重复钉;不同线到同一障碍仍会被钉,合并交给碰撞机制。
        int nseg = network->Nsegs_local;
        for (int s = 0; s < nseg; s++) {
            int n1 = network->segs[s].n1;
            int n2 = network->segs[s].n2;

            Vec3 ng = network->segs[s].plane;
            double ngn = ng.norm();
            if (ngn < 1e-10) continue;          // 无滑移面 -> 跳过
            ng = (1.0 / ngn) * ng;

            // 步初 -> 当前 段位置(o 优先取 xold 记录;当步新生节点退回速度反推)
            Vec3 p1 = network->nodes[n1].pos;
            Vec3 p2 = network->cell.pbc_position(p1, network->nodes[n2].pos);
            Vec3 o1 = (n1 < nxold) ? network->cell.pbc_position(p1, xold(n1))
                                   : p1 - dt * network->nodes[n1].v;
            Vec3 o2 = (n2 < nxold) ? network->cell.pbc_position(p2, xold(n2))
                                   : p2 - dt * network->nodes[n2].v;

            for (int j = 0; j < Nobs; j++) {
                // 只捕获切过型点障碍;Orowan 球归 handle_orowan,不插钉
                if (system->obstacles[j].type != OBSTACLE_BREAKAWAY) continue;

                // endpoint guard: 同段相邻臂已钉此障碍则跳过
                if (network->nodes[n1].sphere_id == j ||
                    network->nodes[n2].sphere_id == j) continue;

                Vec3   C = network->cell.pbc_position(p1, system->obstacles[j].center);
                double R = system->obstacles[j].radius;

                // 垂直门:障碍到该段滑移面的距离 < R?
                double h = dot(C - p1, ng);
                if (h * h >= R * R) continue;
                Vec3 Cproj = C - h * ng;        // 障碍投影到滑移面

                // 本步扫过四边形 [o1,o2,p2,p1] 是否包含 Cproj
                bool hit = point_in_tri(o1, o2, p2, Cproj, ng) ||
                           point_in_tri(o1, p2, p1, Cproj, ng);
                if (!hit) continue;

                // ---- 回退到接触时刻 τ(办法1):端点也退回,消掉过冲尖 ----
                // 段随步线性运动 a(τ)=o1+τ(p1-o1), b(τ)=o2+τ(p2-o2)。
                // 令 Cproj 落在动段直线上:f(τ)=dot(cross(b-a, Cproj-a), ng)=0,
                // 展开为 c0+c1τ+c2τ^2=0,取 [0,1] 内(优先较小=首次接触)的根。
                Vec3 U0 = o2 - o1;
                Vec3 U1 = (p2 - o2) - (p1 - o1);
                Vec3 W0 = Cproj - o1;
                Vec3 W1 = p1 - o1;
                double c0 =  dot(cross(U0, W0), ng);
                double c1 =  dot(cross(U1, W0), ng) - dot(cross(U0, W1), ng);
                double c2 = -dot(cross(U1, W1), ng);
                double tau = 1.0;                          // 兜底:解不出则不回退(= 旧行为)
                if (fabs(c2) < 1e-20) {
                    if (fabs(c1) > 1e-20) {
                        double t = -c0 / c1;
                        if (t >= 0.0 && t <= 1.0) tau = t;
                    }
                } else {
                    double disc = c1 * c1 - 4.0 * c2 * c0;
                    if (disc >= 0.0) {
                        double sq = sqrt(disc);
                        double r1 = (-c1 - sq) / (2.0 * c2);
                        double r2 = (-c1 + sq) / (2.0 * c2);
                        double best = 2.0;
                        if (r1 >= -1e-6 && r1 <= 1.000001 && r1 < best) best = r1;
                        if (r2 >= -1e-6 && r2 <= 1.000001 && r2 < best) best = r2;
                        if (best <= 1.000001) {
                            if (best < 0.0) best = 0.0;
                            if (best > 1.0) best = 1.0;
                            tau = best;
                        }
                    }
                }
                // 端点回退到接触构型(增量式,PBC 安全):pos -= (1-τ)·(p-o)
                // p-o 为本步真实位移(o 来自 xold 时不依赖速度假设)
                network->nodes[n1].pos = network->nodes[n1].pos - (1.0 - tau) * (p1 - o1);
                network->nodes[n2].pos = network->nodes[n2].pos - (1.0 - tau) * (p2 - o2);

                int nnew = network->split_seg(s, Cproj);   // 钉在障碍投影点 C_proj(接触时刻的动段上)
                network->nodes[nnew].constraint = PINNED_NODE;
                network->nodes[nnew].sphere_id  = j;
                ncap++;
                break;                                     // 本段本轮只钉一次
            }
        }

        // ---- Phase B: release ----
        int nnode = network->Nnodes_local;
        for (int i = 0; i < nnode; i++) {
            if (network->nodes[i].constraint != PINNED_NODE) continue;
            int oid = network->nodes[i].sphere_id;
            if (oid < 0) continue;              // BC pin, not managed by this mechanism
            int nc = network->conn[i].num;
            if (nc < 2) continue;

            // R = | sum of unit arm tangents |
            Vec3 pi = network->nodes[i].pos;
            Vec3 sum(0.0);
            for (int k = 0; k < nc; k++) {
                int nbr = network->conn[i].node[k];
                Vec3 t = network->cell.pbc_position(pi, network->nodes[nbr].pos) - pi;
                double tn = t.norm();
                if (tn > 1e-10) sum = sum + (1.0 / tn) * t;
            }

            double Rmag   = sum.norm();
            double phi_c  = system->obstacles[oid].phi_crit;
            double thresh = 2.0 * cos(0.5 * phi_c);
            if (Rmag > thresh) {
                network->nodes[i].constraint = UNCONSTRAINED;
                network->nodes[i].sphere_id  = -1;
                nrel++;
                // 2-arm: R = 2 cos(phi/2) -> phi = 2 acos(R/2)
                double phi = (Rmag < 2.0) ? 2.0 * acos(0.5 * Rmag) : 0.0;
                printf("[Breakaway] depin: R=%.3f  phi=%.1f deg (crit=%.1f)\n",
                       Rmag, phi * 180.0 / M_PI, phi_c * 180.0 / M_PI);
            } else {
                npin++;
            }
        }

        if (ncap || nrel)
            printf("[Breakaway] captured=%d released=%d pinned=%d\n", ncap, nrel, npin);
    }

    /*-----------------------------------------------------------------------
     *  handle  (overrides CollisionRetroactive::handle)
     *---------------------------------------------------------------------*/
    void handle(System* system) override
    {
        // 1. Standard dislocation-dislocation retroactive collision.
        CollisionRetroactive::handle(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].start();

        // 2. Breakaway (cutting) capture/release on point obstacles (serial).
        handle_breakaway(system);

        // 3. Orowan sphere-surface enforcement.
        handle_orowan(system);

        // 4. Twin-plane crossing check: project back any node that crossed.
        handle_twin_wall(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
