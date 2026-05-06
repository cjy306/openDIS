/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    CollisionOrowan
 *    Implements Orowan bypass for spherical precipitates and
 *    twin-boundary blocking for planar obstacles.
 *
 *    Parallelized with Kokkos:
 *      - handle_orowan: parallel_for over nodes to enforce sphere-surface
 *        constraints on the device network.
 *      - handle_twin_detect: parallel_for over segments — if a segment
 *        straddles a twin plane (d1*d2 < 0), project the closer endpoint
 *        back onto the plane and mark it TWIN_SURFACE.
 *      - handle_twin_snap: parallel_for over nodes — snap existing
 *        TWIN_SURFACE nodes back onto their plane (drift correction only).
 *    post_remesh calls both handle_twin_snap (drift) and handle_twin_detect
 *    (catch new crossings created by Topology/Remesh).
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
     *  pre_integrate  (velocity projection for twin boundaries)
     *
     *  Called BEFORE time-integration.  For each node near a twin plane
     *  on the active-zone side, zero the velocity component normal to
     *  the plane if it points toward the vacuum.  This prevents nodes
     *  from ever crossing the plane, eliminating the need for geometric
     *  snap-back (handle_twin_wall / handle_twin_detect) and the
     *  TWIN_SURFACE constraint that blocks coarsening.
     *
     *  Nodes on the vacuum side are NOT blocked — they can return freely.
     *---------------------------------------------------------------------*/
    void pre_integrate(System* system) override
    {
        if (system->planar_obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nnodes  = net->Nnodes_local;
        int Nplanes = (int)system->planar_obstacles.size();

        Kokkos::View<PlanarObstacle*, T_memory_space> d_planes("d_planes_vel", Nplanes);
        auto h_planes = Kokkos::create_mirror_view(d_planes);
        for (int j = 0; j < Nplanes; j++) h_planes(j) = system->planar_obstacles[j];
        Kokkos::deep_copy(d_planes, h_planes);

        auto nodes = net->get_nodes();
        auto cell  = net->cell;
        Vec3 box_center = cell.center();
        double threshold = 2.0 * system->params.rann;

        // Debug: count blocked nodes (use Kokkos::View for atomic reduction)
        Kokkos::View<int, T_memory_space> d_count("d_count");
        Kokkos::deep_copy(d_count, 0);

        Kokkos::parallel_for("TwinVelProject", Nnodes, KOKKOS_LAMBDA(const int i) {
            // Skip pinned and corner nodes
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos = nodes[i].pos;
            Vec3 vel = nodes[i].v;

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d = dot(pos - point, normal);  // signed distance

                if (fabs(d) > threshold) continue;

                double vn = dot(vel, normal);

                // Determine which side is the active zone
                double center_d = dot(box_center - point, normal);
                int active_sign = (center_d > 0.0) ? 1 : -1;

                // Block normal velocity if node is on the active side
                // AND velocity points toward the vacuum
                if (d * active_sign >= 0.0 && vn * active_sign < 0.0) {
                    nodes[i].v = vel - vn * normal;  // zero normal component
                    Kokkos::atomic_inc(&d_count());
                    return;  // one plane per node per step
                }
            }
        });
        Kokkos::fence();

        auto h_count = Kokkos::create_mirror_view(d_count);
        Kokkos::deep_copy(h_count, d_count);
        if (h_count() > 0)
            printf("[TwinVelProj] step blocked %d node(s)\n", h_count());
    }

    /*-----------------------------------------------------------------------
     *  handle_orowan  (Kokkos parallel)
     *  Runs entirely on the device network.  No topology changes.
     *---------------------------------------------------------------------*/
    void handle_orowan(System* system)
    {
        if (system->obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nnodes = net->Nnodes_local;
        int Nobs = (int)system->obstacles.size();

        // Copy spherical obstacles to device memory
        Kokkos::View<SphericalObstacle*, T_memory_space> d_obs("d_obs", Nobs);
        auto h_obs = Kokkos::create_mirror_view(d_obs);
        for (int j = 0; j < Nobs; j++) h_obs(j) = system->obstacles[j];
        Kokkos::deep_copy(d_obs, h_obs);

        auto nodes = net->get_nodes();

        Kokkos::parallel_for("OrowanNodes", Nnodes, KOKKOS_LAMBDA(const int i) {
            Vec3 pos = nodes[i].pos;

            for (int j = 0; j < Nobs; j++) {
                Vec3   d     = pos - d_obs(j).center;
                double dist2 = d.norm2();
                double r2    = d_obs(j).radius * d_obs(j).radius;

                if (dist2 < r2) {
                    // Node inside sphere: push to surface.
                    double dist   = sqrt(dist2);
                    Vec3   normal = (dist > 1e-10) ? (1.0/dist) * d
                                                   : Vec3(0.0, 0.0, 1.0);
                    nodes[i].pos            = d_obs(j).center + d_obs(j).radius * normal;
                    nodes[i].constraint     = SPHERE_SURFACE;
                    nodes[i].sphere_id      = d_obs(j).id;
                    nodes[i].sphere_normal  = normal;

                } else if (nodes[i].constraint == SPHERE_SURFACE &&
                           nodes[i].sphere_id  == d_obs(j).id) {
                    double dist = sqrt(dist2);
                    if (dist > d_obs(j).radius * 1.05) {
                        // Released (topology handled bypass).
                        nodes[i].constraint     = UNCONSTRAINED;
                        nodes[i].sphere_id      = -1;
                        nodes[i].sphere_normal  = Vec3(0.0);
                    } else {
                        // Refresh outward normal.
                        nodes[i].sphere_normal =
                            (dist > 1e-10) ? (1.0/dist) * d : Vec3(0.0, 0.0, 1.0);
                    }
                }
            }
        });
        Kokkos::fence();
    }

    /*-----------------------------------------------------------------------
     *  handle_twin_detect  (Kokkos parallel over segments)
     *
     *  Per-segment crossing detection: for each segment, check whether the
     *  two endpoints lie on opposite sides of a twin plane (d1*d2 < 0).
     *  If so, project the CLOSER endpoint (smaller |d|) back onto the plane
     *  and mark it TWIN_SURFACE.
     *
     *  This avoids using xold (which becomes invalid after Topology/Remesh)
     *  and minimizes geometry distortion by only moving the node that just
     *  barely crossed.
     *---------------------------------------------------------------------*/
    void handle_twin_detect(System* system)
    {
        if (system->planar_obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nsegs   = net->Nsegs_local;
        int Nplanes = (int)system->planar_obstacles.size();
        double maxseg = system->params.maxseg;

        // Copy planar obstacles to device memory
        Kokkos::View<PlanarObstacle*, T_memory_space> d_planes("d_planes", Nplanes);
        auto h_planes = Kokkos::create_mirror_view(d_planes);
        for (int j = 0; j < Nplanes; j++) h_planes(j) = system->planar_obstacles[j];
        Kokkos::deep_copy(d_planes, h_planes);

        auto nodes = net->get_nodes();
        auto segs  = net->get_segs();
        auto cell  = net->cell;

        Kokkos::parallel_for("TwinDetect", Nsegs, KOKKOS_LAMBDA(const int s) {
            int i1 = segs[s].n1;
            int i2 = segs[s].n2;

            bool tw1 = (nodes[i1].constraint == TWIN_SURFACE);
            bool tw2 = (nodes[i2].constraint == TWIN_SURFACE);

            // Skip segments where BOTH endpoints are already twin pivots
            if (tw1 && tw2) return;

            Vec3 pos1 = nodes[i1].pos;
            Vec3 pos2 = cell.pbc_position(pos1, nodes[i2].pos);

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d1 = dot(pos1 - point, normal);
                double d2 = dot(pos2 - point, normal);

                // Case 1: Neither endpoint is TWIN_SURFACE — standard straddling check
                if (!tw1 && !tw2) {
                    if (d1 * d2 < 0.0) {
                        double abs_d1 = fabs(d1);
                        double abs_d2 = fabs(d2);
                        if (abs_d1 <= abs_d2) {
                            if (abs_d1 < maxseg) {
                                nodes[i1].pos        = pos1 - d1 * normal;
                                nodes[i1].constraint = TWIN_SURFACE;
                                nodes[i1].twin_id    = d_planes(j).id;
                                nodes[i1].twin_normal = normal;
                            }
                        } else {
                            if (abs_d2 < maxseg) {
                                Vec3 shift = pos2 - nodes[i2].pos;
                                Vec3 projected = pos2 - d2 * normal;
                                nodes[i2].pos        = projected - shift;
                                nodes[i2].constraint = TWIN_SURFACE;
                                nodes[i2].twin_id    = d_planes(j).id;
                                nodes[i2].twin_normal = normal;
                            }
                        }
                        return;
                    }
                }

                // Case 2: One endpoint is TWIN_SURFACE on this plane,
                // the other is free. Check strict straddling (d1*d2 < 0).
                // Since the twin node is snapped to the plane (d ≈ 0),
                // d1*d2 < 0 can only trigger if the free node has clearly
                // crossed to the opposite side.  This is a conservative
                // fallback — handle_twin_wall catches most crossings.
                if (tw1 && !tw2 && nodes[i1].twin_id == d_planes(j).id) {
                    if (d1 * d2 < 0.0 && fabs(d2) < maxseg) {
                        Vec3 shift = pos2 - nodes[i2].pos;
                        Vec3 projected = pos2 - d2 * normal;
                        nodes[i2].pos        = projected - shift;
                        nodes[i2].constraint = TWIN_SURFACE;
                        nodes[i2].twin_id    = d_planes(j).id;
                        nodes[i2].twin_normal = normal;
                        return;
                    }
                }

                if (tw2 && !tw1 && nodes[i2].twin_id == d_planes(j).id) {
                    if (d1 * d2 < 0.0 && fabs(d1) < maxseg) {
                        nodes[i1].pos        = pos1 - d1 * normal;
                        nodes[i1].constraint = TWIN_SURFACE;
                        nodes[i1].twin_id    = d_planes(j).id;
                        nodes[i1].twin_normal = normal;
                        return;
                    }
                }
            }
        });
        Kokkos::fence();
    }

    /*-----------------------------------------------------------------------
     *  handle_twin_wall  (Kokkos parallel over nodes)
     *
     *  Per-node crossing detection using xold.  For each free node,
     *  check if it has crossed any twin plane between its old position
     *  (xold) and its current position.  If so, project it back onto
     *  the plane and mark it TWIN_SURFACE.
     *
     *  MUST be called while system->xold is still valid (i.e. after
     *  integration but before topology/remesh which invalidate xold).
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
        auto xold  = system->xold;
        auto cell  = net->cell;

        Kokkos::parallel_for("TwinWall", Nnodes, KOKKOS_LAMBDA(const int i) {
            // Skip nodes already constrained (twin, sphere, pinned)
            if (nodes[i].constraint != UNCONSTRAINED) return;

            Vec3 pos_new = nodes[i].pos;
            Vec3 pos_old = cell.pbc_position(pos_new, xold(i));

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d_old = dot(pos_old - point, normal);
                double d_new = dot(pos_new - point, normal);

                if (d_old * d_new < 0.0) {
                    // Node crossed the plane: project back onto the plane
                    nodes[i].pos         = pos_new - d_new * normal;
                    nodes[i].constraint  = TWIN_SURFACE;
                    nodes[i].twin_id     = d_planes(j).id;
                    nodes[i].twin_normal = normal;
                    return;  // one crossing per node per step
                }
            }
        });
        Kokkos::fence();
    }

    /*-----------------------------------------------------------------------
     *  handle_twin_snap  (Kokkos parallel over nodes)
     *
     *  For existing TWIN_SURFACE nodes:
     *  1. Check release condition: if all neighbor nodes are on the SAME
     *     side of the twin plane, the dislocation has bypassed/retracted
     *     and the node can be released to UNCONSTRAINED.
     *  2. Otherwise, snap the node back onto its plane (drift correction).
     *
     *  Safe to call after Topology/Remesh (no xold dependency).
     *---------------------------------------------------------------------*/
    void handle_twin_snap(System* system)
    {
        if (system->planar_obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nnodes  = net->Nnodes_local;
        int Nplanes = (int)system->planar_obstacles.size();

        Kokkos::View<PlanarObstacle*, T_memory_space> d_planes("d_planes", Nplanes);
        auto h_planes = Kokkos::create_mirror_view(d_planes);
        for (int j = 0; j < Nplanes; j++) h_planes(j) = system->planar_obstacles[j];
        Kokkos::deep_copy(d_planes, h_planes);

        auto nodes = net->get_nodes();
        auto conn  = net->get_conn();
        auto cell  = net->cell;

        Kokkos::parallel_for("TwinSnap", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint != TWIN_SURFACE) return;

            int tid = nodes[i].twin_id;
            int plane_idx = -1;
            for (int j = 0; j < Nplanes; j++) {
                if (d_planes(j).id == tid) { plane_idx = j; break; }
            }
            if (plane_idx < 0) {
                // Orphaned twin node (plane removed?): release it
                nodes[i].constraint = UNCONSTRAINED;
                nodes[i].twin_id = -1;
                return;
            }

            Vec3 normal = d_planes(plane_idx).normal;
            Vec3 point  = d_planes(plane_idx).point;

            // Release check: if this is a 2-arm node and all neighbors
            // are on the same side of the plane, the dislocation is no
            // longer straddling → release the node.
            int nconn = conn[i].num;
            if (nconn == 2) {
                bool all_same_side = true;
                double first_sign = 0.0;
                for (int k = 0; k < nconn; k++) {
                    int nbr = conn[i].node[k];
                    // Skip neighbors that are also TWIN_SURFACE on same plane
                    if (nodes[nbr].constraint == TWIN_SURFACE &&
                        nodes[nbr].twin_id == tid) continue;
                    Vec3 nbr_pos = cell.pbc_position(nodes[i].pos, nodes[nbr].pos);
                    double dn = dot(nbr_pos - point, normal);
                    if (fabs(dn) < 1e-10) continue;  // on the plane
                    if (first_sign == 0.0) {
                        first_sign = dn;
                    } else if (first_sign * dn < 0.0) {
                        all_same_side = false;
                        break;
                    }
                }
                // Only release if we actually checked at least one
                // neighbor off the plane and they were all on the same side
                if (all_same_side && first_sign != 0.0) {
                    nodes[i].constraint = UNCONSTRAINED;
                    nodes[i].twin_id = -1;
                    return;
                }
            }

            // Snap to plane (drift correction)
            double d = dot(nodes[i].pos - point, normal);
            nodes[i].pos = nodes[i].pos - d * normal;
        });
        Kokkos::fence();
    }

    /*-----------------------------------------------------------------------
     *  handle  (overrides CollisionRetroactive::handle)
     *---------------------------------------------------------------------*/
    void handle(System* system) override
    {
        // 1. Standard dislocation-dislocation retroactive collision first.
        CollisionRetroactive::handle(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].start();

        // 2. Orowan sphere-surface enforcement (parallel on device).
        handle_orowan(system);

        // 3. Per-node wall detection using xold (most reliable — catches
        //    any free node that crossed a twin plane during integration).
        //    Must be called while xold is still valid.
        handle_twin_wall(system);

        // 4. Per-segment straddling detection (catches any remaining
        //    crossings not caught by wall detection, e.g. segments
        //    where both nodes were on the same side but one has now
        //    been projected onto the plane by handle_twin_wall).
        handle_twin_detect(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    /*-----------------------------------------------------------------------
     *  post_remesh  (called after Remesh in driver.cpp)
     *  1. Snap existing TWIN_SURFACE nodes back onto their plane (drift).
     *  2. Detect NEW crossings created by Topology/Remesh.
     *     handle_twin_detect is xold-free (per-segment), so safe here.
     *---------------------------------------------------------------------*/
    void post_remesh(System* system) override
    {
        handle_twin_snap(system);
        handle_twin_detect(system);
    }

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
