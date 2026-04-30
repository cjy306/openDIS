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

                // Case 2: One endpoint is TWIN_SURFACE, the other is free.
                // The twin node is on the plane (d ≈ 0). If the free node
                // has crossed to the opposite side, project it back onto
                // the plane and mark it TWIN_SURFACE.
                if (tw1 && !tw2 && nodes[i1].twin_id == d_planes(j).id) {
                    // i1 is on the plane; check if i2 crossed
                    // d1 ≈ 0, so use sign of d2 vs sign of original side.
                    // Since i1 is on the plane, any |d2| very small means
                    // i2 is near the plane — project it.
                    if (fabs(d2) < maxseg && d1 * d2 < 0.0) {
                        Vec3 shift = pos2 - nodes[i2].pos;
                        Vec3 projected = pos2 - d2 * normal;
                        nodes[i2].pos        = projected - shift;
                        nodes[i2].constraint = TWIN_SURFACE;
                        nodes[i2].twin_id    = d_planes(j).id;
                        nodes[i2].twin_normal = normal;
                        return;
                    }
                    // Also catch: d1 ≈ 0 so d1*d2 ≈ 0 (not < 0).
                    // Use a small threshold: if i2 has crossed past the plane
                    // (d2 on opposite side from where the segment "came from"),
                    // we detect this by checking that |d2| is small but nonzero
                    // and on the opposite side of the plane normal from the
                    // segment interior direction.
                    double seg_dot = dot(pos2 - pos1, normal);
                    if (fabs(d2) < maxseg * 0.5 && seg_dot * d2 > 0.0) {
                        // i2 has drifted past the plane
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
                    // i2 is on the plane; check if i1 crossed
                    if (fabs(d1) < maxseg && d1 * d2 < 0.0) {
                        nodes[i1].pos        = pos1 - d1 * normal;
                        nodes[i1].constraint = TWIN_SURFACE;
                        nodes[i1].twin_id    = d_planes(j).id;
                        nodes[i1].twin_normal = normal;
                        return;
                    }
                    double seg_dot = dot(pos1 - pos2, normal);
                    if (fabs(d1) < maxseg * 0.5 && seg_dot * d1 > 0.0) {
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
     *  Drift correction only: snap existing TWIN_SURFACE nodes back onto
     *  their plane.  No new crossing detection — safe to call after
     *  Topology/Remesh when xold is invalid.
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

        Kokkos::parallel_for("TwinSnap", Nnodes, KOKKOS_LAMBDA(const int i) {
            if (nodes[i].constraint != TWIN_SURFACE) return;

            int tid = nodes[i].twin_id;
            for (int j = 0; j < Nplanes; j++) {
                if (d_planes(j).id == tid) {
                    double d = dot(nodes[i].pos - d_planes(j).point, d_planes(j).normal);
                    nodes[i].pos = nodes[i].pos - d * d_planes(j).normal;
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
