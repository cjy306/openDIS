/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    CollisionOrowan
 *    Implements Orowan bypass for spherical precipitates and
 *    twin-boundary blocking for planar obstacles.
 *
 *    Twin boundary strategy:
 *    - pre_integrate: velocity-based blocking prevents crossing
 *    - handle_twin_wall: safety-net projection to d=0 for crossers
 *    - Projected nodes marked TWIN_SURFACE so CollisionRetroactive
 *      skips them (prevents coplanar collision pile-up / MAX_CONN)
 *    - Nodes that move away from the plane are unmarked automatically
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
     *  Called BEFORE time-integration.  For each node on the active side
     *  whose normal velocity would carry it past the twin plane within
     *  safety * dt, zero the normal velocity component.
     *
     *  Also unmarks TWIN_SURFACE nodes that have drifted away from the
     *  plane (d > rann), restoring them to UNCONSTRAINED.
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
        double dt = system->realdt;
        if (dt <= 0.0) dt = system->params.nextdt;
        double safety = 2.0;
        double rann = system->params.rann;

        Kokkos::parallel_for("TwinVelProject", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos = nodes[i].pos;
            Vec3 vel = nodes[i].v;

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d = dot(pos - point, normal);

                double center_d = dot(box_center - point, normal);
                int active_sign = (center_d > 0.0) ? 1 : -1;

                // Unmark TWIN_SURFACE nodes that moved away from the plane
                if (nodes[i].constraint == TWIN_SURFACE &&
                    nodes[i].twin_id == j) {
                    if (fabs(d) > rann) {
                        nodes[i].constraint = UNCONSTRAINED;
                        nodes[i].twin_id = -1;
                    }
                }

                // Block velocity if on active side, moving toward vacuum,
                // and would reach the plane within safety * dt
                double vn = dot(vel, normal);
                if (d * active_sign >= 0.0 && vn * active_sign < 0.0) {
                    double abs_d = fabs(d);
                    double displacement = fabs(vn) * dt * safety;
                    if (abs_d < displacement) {
                        nodes[i].v = vel - vn * normal;
                        return;
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
     *  handle_twin_wall  (Kokkos parallel over nodes)
     *
     *  Per-node crossing detection using xold.  If a node crossed a twin
     *  plane, project it back to d=0 and mark it TWIN_SURFACE.
     *  The TWIN_SURFACE constraint causes CollisionRetroactive to skip
     *  collision detection for segments involving this node, preventing
     *  coplanar collision pile-up and MAX_CONN overflow.
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
                    // Project to d=0 and mark TWIN_SURFACE
                    nodes[i].pos = pos_new - d_new * normal;
                    nodes[i].constraint = TWIN_SURFACE;
                    nodes[i].twin_id = j;
                    nodes[i].twin_normal = normal;
                    return;
                }
            }
        });
        Kokkos::fence();
    }

    /*-----------------------------------------------------------------------
     *  handle_twin_snap  (Kokkos parallel over nodes)
     *
     *  Safety-net: for any node on the vacuum side within threshold,
     *  project to d=0 and mark TWIN_SURFACE.
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
        auto cell  = net->cell;
        Vec3 box_center = cell.center();
        double threshold = system->params.minseg;

        Kokkos::parallel_for("TwinSnap", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos = nodes[i].pos;
            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d = dot(pos - point, normal);

                if (fabs(d) > threshold) continue;

                double center_d = dot(box_center - point, normal);
                double sign = (center_d > 0.0) ? 1.0 : -1.0;

                // Snap vacuum-side node to d=0 and mark TWIN_SURFACE
                if (d * sign < 0.0) {
                    nodes[i].pos = pos - d * normal;
                    nodes[i].constraint = TWIN_SURFACE;
                    nodes[i].twin_id = j;
                    nodes[i].twin_normal = normal;
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
        // 1. Standard dislocation-dislocation retroactive collision.
        //    CollisionRetroactive skips TWIN_SURFACE nodes automatically.
        CollisionRetroactive::handle(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].start();

        // 2. Orowan sphere-surface enforcement.
        handle_orowan(system);

        // 3. Per-node wall detection using xold.
        //    Projects crossers to d=0 and marks them TWIN_SURFACE.
        handle_twin_wall(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    /*-----------------------------------------------------------------------
     *  post_remesh  (called after Remesh in driver.cpp)
     *---------------------------------------------------------------------*/
    void post_remesh(System* system) override
    {
        handle_twin_snap(system);
    }

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
