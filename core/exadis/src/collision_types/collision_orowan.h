/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    CollisionOrowan
 *    Implements Orowan bypass for spherical precipitates and
 *    twin-boundary blocking for planar obstacles.
 *
 *    Twin boundary strategy: projection without constraint marking.
 *    Nodes that cross a twin plane are projected back onto it but
 *    remain UNCONSTRAINED — remesh coarsening can merge them freely.
 *    Velocity projection (pre_integrate) prevents crossing before
 *    integration; wall/detect/snap provide safety-net projection after.
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
        // Velocity-based blocking: use upcoming timestep to determine
        // whether each node would cross the plane in this step.
        // Only blocks nodes that would actually cross — no fixed distance threshold.
        double dt = system->realdt;
        if (dt <= 0.0) dt = system->params.nextdt;
        double safety = 2.0;  // safety factor

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

                double vn = dot(vel, normal);

                // Determine which side is the active zone
                double center_d = dot(box_center - point, normal);
                int active_sign = (center_d > 0.0) ? 1 : -1;

                // Block if: node on active side, velocity toward vacuum,
                // AND would reach the plane within safety * dt
                if (d * active_sign >= 0.0 && vn * active_sign < 0.0) {
                    double abs_d = fabs(d);
                    double displacement = fabs(vn) * dt * safety;
                    if (abs_d < displacement) {
                        nodes[i].v = vel - vn * normal;
                        Kokkos::atomic_inc(&d_count());
                        return;  // one plane per node per step
                    }
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
     *
     *  For each non-pinned node, check if it has entered any spherical
     *  obstacle.  If inside, push it back onto the obstacle surface.
     *  No constraint marking — nodes remain UNCONSTRAINED so remesh
     *  coarsening can merge them freely.
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
            if (nodes[i].constraint == PINNED_NODE ||
                nodes[i].constraint == CORNER_NODE) return;

            Vec3 pos = nodes[i].pos;

            for (int j = 0; j < Nobs; j++) {
                Vec3   d     = pos - d_obs(j).center;
                double dist2 = d.norm2();
                double r2    = d_obs(j).radius * d_obs(j).radius;

                if (dist2 < r2) {
                    // Node inside sphere: push to surface, no marking.
                    double dist   = sqrt(dist2);
                    Vec3   normal = (dist > 1e-10) ? (1.0/dist) * d
                                                   : Vec3(0.0, 0.0, 1.0);
                    nodes[i].pos = d_obs(j).center + d_obs(j).radius * normal;
                    return;  // one sphere per node per step
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
     *  If so, project the CLOSER endpoint (smaller |d|) back onto the plane.
     *  No constraint marking — projected nodes remain UNCONSTRAINED.
     *
     *  xold-free (safe after Topology/Remesh).
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

            Vec3 pos1 = nodes[i1].pos;
            Vec3 pos2 = cell.pbc_position(pos1, nodes[i2].pos);

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d1 = dot(pos1 - point, normal);
                double d2 = dot(pos2 - point, normal);

                // Segment straddles plane: project closer endpoint (no marking)
                if (d1 * d2 < 0.0) {
                    double abs_d1 = fabs(d1);
                    double abs_d2 = fabs(d2);
                    if (abs_d1 <= abs_d2) {
                        if (abs_d1 < maxseg) {
                            nodes[i1].pos = pos1 - d1 * normal;
                        }
                    } else {
                        if (abs_d2 < maxseg) {
                            Vec3 shift = pos2 - nodes[i2].pos;
                            Vec3 projected = pos2 - d2 * normal;
                            nodes[i2].pos = projected - shift;
                        }
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
     *  Per-node crossing detection using xold.  For each non-pinned node,
     *  check if it has crossed any twin plane between its old position
     *  (xold) and its current position.  If so, project it back onto
     *  the plane.  No constraint marking — nodes remain UNCONSTRAINED.
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
                    // Node crossed the plane: project back onto the plane
                    nodes[i].pos = pos_new - d_new * normal;
                    return;  // one crossing per node per step
                }
            }
        });
        Kokkos::fence();
    }

    /*-----------------------------------------------------------------------
     *  handle_twin_snap  (Kokkos parallel over nodes)
     *
     *  Safety-net projection: for any node on the vacuum side of a twin
     *  plane (within threshold), project it back onto the plane.
     *  No constraint marking — nodes remain UNCONSTRAINED so remesh
     *  coarsening can merge them freely.
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

                // Determine active/vacuum side
                double center_d = dot(box_center - point, normal);
                int active_sign = (center_d > 0.0) ? 1 : -1;

                // Snap node if on vacuum side (safety net projection)
                if (d * active_sign < 0.0) {
                    nodes[i].pos = pos - d * normal;
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

        // 3. Per-node wall detection using xold (most reliable — catches
        //    any free node that crossed a twin plane during integration).
        //    Must be called while xold is still valid.
        handle_twin_wall(system);

        // handle_twin_detect removed: it caused excessive projections
        // and collision pile-ups at the twin plane. handle_twin_wall
        // catches actual crossings; pre_integrate prevents most of them.

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    /*-----------------------------------------------------------------------
     *  post_remesh  (called after Remesh in driver.cpp)
     *  1. Vacuum-side safety-net projection (drift correction).
     *  2. Detect NEW segment crossings created by Topology/Remesh.
     *     handle_twin_detect is xold-free (per-segment), so safe here.
     *---------------------------------------------------------------------*/
    void post_remesh(System* system) override
    {
        handle_twin_snap(system);
    }

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
