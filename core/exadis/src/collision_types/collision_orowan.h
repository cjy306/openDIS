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

            // Skip segments where either endpoint is already a twin pivot
            if (nodes[i1].constraint == TWIN_SURFACE ||
                nodes[i2].constraint == TWIN_SURFACE) return;

            Vec3 pos1 = nodes[i1].pos;
            Vec3 pos2 = cell.pbc_position(pos1, nodes[i2].pos);

            for (int j = 0; j < Nplanes; j++) {
                Vec3   normal = d_planes(j).normal;
                Vec3   point  = d_planes(j).point;
                double d1 = dot(pos1 - point, normal);
                double d2 = dot(pos2 - point, normal);

                if (d1 * d2 < 0.0) {
                    // Segment straddles the plane.
                    // Project the closer endpoint (smaller |d|).
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
                            // Undo PBC shift when writing back
                            Vec3 shift = pos2 - nodes[i2].pos;
                            Vec3 projected = pos2 - d2 * normal;
                            nodes[i2].pos        = projected - shift;
                            nodes[i2].constraint = TWIN_SURFACE;
                            nodes[i2].twin_id    = d_planes(j).id;
                            nodes[i2].twin_normal = normal;
                        }
                    }
                    return;  // One crossing per segment per step
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

        // 3. Twin-boundary crossing detection + projection (parallel on device).
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
