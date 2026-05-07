/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    CollisionOrowan
 *    Implements Orowan bypass for spherical precipitates and
 *    twin-boundary blocking for planar obstacles.
 *
 *    Twin boundary strategy: FORCE-BASED repulsion.
 *    An exponential repulsive force is added to nodes approaching
 *    a twin plane from the active side.  The force is purely normal
 *    to the plane, so tangential motion (gliding along the boundary)
 *    is unrestricted.  This works within the standard DDD force →
 *    mobility → integration pipeline without any position or velocity
 *    manipulation.
 *
 *    handle_twin_wall is kept as a lightweight safety net in case
 *    a node still overshoots the plane.
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
     *  add_obstacle_force  (called between force->compute and mobility)
     *
     *  Adds an exponential repulsive force for each node near a twin plane
     *  on the active side.  Force acts purely in the plane-normal direction:
     *
     *      F_repel = F0 * exp(-d / lambda) * n_active
     *
     *  where d is the distance to the plane, lambda is the decay length,
     *  and F0 = mu * b^2 / lambda provides a physically-scaled magnitude.
     *---------------------------------------------------------------------*/
    void add_obstacle_force(System* system) override
    {
        if (system->planar_obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nnodes  = net->Nnodes_local;
        int Nplanes = (int)system->planar_obstacles.size();

        Kokkos::View<PlanarObstacle*, T_memory_space> d_planes("d_planes_force", Nplanes);
        auto h_planes = Kokkos::create_mirror_view(d_planes);
        for (int j = 0; j < Nplanes; j++) h_planes(j) = system->planar_obstacles[j];
        Kokkos::deep_copy(d_planes, h_planes);

        auto nodes = net->get_nodes();
        auto cell  = net->cell;
        Vec3 box_center = cell.center();

        // Force parameters
        double mu      = system->params.MU;
        double burgmag = system->params.burgmag;
        double lambda  = system->params.minseg;  // decay length (50b)
        double F0      = 10.0 * mu * burgmag * burgmag / lambda;  // peak force

        Kokkos::parallel_for("TwinRepulsion", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos = nodes[i].pos;

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d = dot(pos - point, normal);  // signed distance

                // Determine active side
                double center_d = dot(box_center - point, normal);
                double active_sign = (center_d > 0.0) ? 1.0 : -1.0;

                double d_active = d * active_sign;  // positive on active side
                double abs_d = fabs(d);

                if (abs_d < 5.0 * lambda) {
                    if (d_active > 0.0) {
                        // Active side: exponential repulsion
                        double F_mag = F0 * exp(-d_active / lambda);
                        Vec3 F_repel = F_mag * active_sign * normal;
                        Kokkos::atomic_add(&nodes[i].f, F_repel);
                    } else {
                        // Crossed to wrong side: strong restoring force
                        double F_mag = F0 * (1.0 + fabs(d_active) / lambda);
                        Vec3 F_restore = F_mag * active_sign * normal;
                        Kokkos::atomic_add(&nodes[i].f, F_restore);
                    }
                }
            }
        });
        Kokkos::fence();
    }

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
     *  handle_twin_wall  (safety net — Kokkos parallel over nodes)
     *
     *  Lightweight fallback: if a node still crosses a twin plane despite
     *  the repulsive force (e.g. very large dt), project it back to d=0.
     *  Should rarely fire with properly tuned force parameters.
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

        Kokkos::View<int, T_memory_space> d_count("d_count_wall");
        Kokkos::deep_copy(d_count, 0);

        Kokkos::parallel_for("TwinWall", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos_new = nodes[i].pos;
            Vec3 pos_old = cell.pbc_position(pos_new, xold(i));

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d_old = dot(pos_old - point, normal);
                double d_new = dot(pos_new - point, normal);

                if (d_old * d_new < 0.0) {
                    nodes[i].pos = pos_new - d_new * normal;
                    Kokkos::atomic_inc(&d_count());
                    return;
                }
            }
        });
        Kokkos::fence();

        auto h_count = Kokkos::create_mirror_view(d_count);
        Kokkos::deep_copy(h_count, d_count);
        if (h_count() > 0)
            printf("[TwinWall] safety-net projected %d node(s)\n", h_count());
    }

    /*-----------------------------------------------------------------------
     *  pre_integrate  (no-op — force-based approach handles everything)
     *---------------------------------------------------------------------*/
    void pre_integrate(System* system) override {}

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

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    /*-----------------------------------------------------------------------
     *  post_remesh  (no-op — force-based approach needs no post-remesh fix)
     *---------------------------------------------------------------------*/
    void post_remesh(System* system) override {}

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
