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

        Kokkos::parallel_for("OrowanNodes", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos = nodes[i].pos;

            for (int j = 0; j < Nobs; j++) {
                Vec3   d     = pos - d_obs(j).center;
                double dist2 = d.norm2();
                double r2    = d_obs(j).radius * d_obs(j).radius;

                if (dist2 < r2) {
                    double dist   = sqrt(dist2);
                    Vec3   normal = (dist > 1e-10) ? (1.0/dist) * d
                                                   : Vec3(0.0, 0.0, 1.0);
                    nodes[i].pos = d_obs(j).center + d_obs(j).radius * normal;
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
     *  handle  (overrides CollisionRetroactive::handle)
     *---------------------------------------------------------------------*/
    void handle(System* system) override
    {
        // 1. Standard dislocation-dislocation retroactive collision.
        CollisionRetroactive::handle(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].start();

        // 2. Orowan sphere-surface enforcement.
        handle_orowan(system);

        // 3. Twin-plane crossing check: project back any node that crossed.
        handle_twin_wall(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
