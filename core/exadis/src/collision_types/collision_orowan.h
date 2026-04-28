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
 *      - handle_twin: parallel_for over nodes — if a node crossed a twin
 *        plane (comparing pos vs xold), project it back onto the plane
 *        and mark it TWIN_SURFACE so the mobility module zeroes the
 *        normal velocity component (node slides on the plane).
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
     *  handle_twin  (fully Kokkos parallel — no topology changes)
     *
     *  For each node, compare current pos with xold to detect whether
     *  the node crossed a twin plane during this time step.  If it did,
     *  project the node back onto the plane and mark it TWIN_SURFACE.
     *  The mobility module then zeroes the normal velocity component,
     *  so the node can slide along the plane but never cross it.
     *
     *  Existing TWIN_SURFACE nodes are snapped back onto the plane
     *  to correct numerical drift.
     *---------------------------------------------------------------------*/
    void handle_twin(System* system)
    {
        if (system->planar_obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nnodes  = net->Nnodes_local;
        int Nplanes = (int)system->planar_obstacles.size();

        // Copy planar obstacles to device memory
        Kokkos::View<PlanarObstacle*, T_memory_space> d_planes("d_planes", Nplanes);
        auto h_planes = Kokkos::create_mirror_view(d_planes);
        for (int j = 0; j < Nplanes; j++) h_planes(j) = system->planar_obstacles[j];
        Kokkos::deep_copy(d_planes, h_planes);

        auto nodes = net->get_nodes();
        auto xold  = system->xold;

        Kokkos::parallel_for("TwinProject", Nnodes, KOKKOS_LAMBDA(const int i) {
            Vec3 pos = nodes[i].pos;

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d_now  = dot(pos - point, normal);

                if (nodes[i].constraint == TWIN_SURFACE) {
                    // Already pinned: snap back onto plane to fix drift.
                    nodes[i].pos = pos - d_now * normal;
                    return;
                }

                // Check if node crossed the plane: old side vs new side.
                double d_old = dot(xold(i) - point, normal);
                if (d_old * d_now < 0.0) {
                    // Crossed — project back onto the plane.
                    nodes[i].pos            = pos - d_now * normal;
                    nodes[i].constraint     = TWIN_SURFACE;
                    nodes[i].twin_id        = d_planes(j).id;
                    nodes[i].twin_normal    = normal;
                    return;
                }
            }
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

        // 3. Twin-boundary node projection (fully parallel on device).
        handle_twin(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    /*-----------------------------------------------------------------------
     *  post_remesh  (called after Remesh in driver.cpp)
     *  Re-run twin enforcement to catch new crossings created by
     *  Topology and Remesh operations.
     *---------------------------------------------------------------------*/
    void post_remesh(System* system) override
    {
        handle_twin(system);
    }

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
