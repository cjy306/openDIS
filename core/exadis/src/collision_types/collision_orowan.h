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
 *      - handle_twin: Phase 1 parallel detection of segment-plane crossings
 *        on the device, Phase 2 serial split_seg on the host to insert
 *        pivot nodes at intersection points.  Pivot nodes are marked
 *        TWIN_SURFACE so that the mobility module projects their velocity
 *        onto the twin plane (sliding but not crossing).
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
     *  TwinCrossingEvent
     *  Stores a detected segment-plane crossing for Phase 2 execution.
     *---------------------------------------------------------------------*/
    struct TwinCrossingEvent {
        int seg_id;
        int plane_id;
        Vec3 cross_pos;
        Vec3 normal;
    };

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
     *  handle_twin  (Kokkos parallel detection + serial split_seg)
     *
     *  Phase 1 (device): for each segment, check whether the two endpoints
     *    lie on opposite sides of a twin plane (d1*d2 < 0).  If so, compute
     *    the intersection point and record a TwinCrossingEvent.
     *
     *  Phase 2 (host/serial): for each event, call split_seg to insert a
     *    new pivot node at the intersection point.  The pivot node is
     *    marked TWIN_SURFACE so the mobility module projects its velocity
     *    onto the plane (slides but cannot cross).
     *
     *  Phase 3 (host/serial): refresh existing TWIN_SURFACE nodes —
     *    snap back to plane if drifted, or release if moved far away
     *    (e.g. after topology merge / Orowan bypass completion).
     *---------------------------------------------------------------------*/
    void handle_twin(System* system)
    {
        if (system->planar_obstacles.empty()) return;

        DeviceDisNet* net = system->get_device_network();
        int Nsegs   = net->Nsegs_local;
        int Nplanes = (int)system->planar_obstacles.size();

        // Copy planar obstacles to device memory
        Kokkos::View<PlanarObstacle*, T_memory_space> d_planes("d_planes", Nplanes);
        auto h_planes = Kokkos::create_mirror_view(d_planes);
        for (int j = 0; j < Nplanes; j++) h_planes(j) = system->planar_obstacles[j];
        Kokkos::deep_copy(d_planes, h_planes);

        // --- Phase 1: parallel crossing detection on device ---
        int max_events = Nsegs;  // upper bound (one crossing per segment)
        Kokkos::View<TwinCrossingEvent*, T_memory_shared> events("twin_events", max_events);
        Kokkos::View<int, T_memory_shared> count("twin_count");

        auto nodes = net->get_nodes();
        auto segs  = net->get_segs();

        Kokkos::parallel_for("TwinDetect", Nsegs, KOKKOS_LAMBDA(const int s) {
            int  n1 = segs[s].n1;
            int  n2 = segs[s].n2;

            // Skip segments where either endpoint is already a twin pivot.
            // Without this check, numerical drift of pivot nodes causes
            // cascading re-splits that explode the node count.
            if (nodes[n1].constraint == TWIN_SURFACE ||
                nodes[n2].constraint == TWIN_SURFACE) return;

            Vec3 p1 = nodes[n1].pos;
            Vec3 p2 = nodes[n2].pos;

            for (int j = 0; j < Nplanes; j++) {
                double d1 = dot(p1 - d_planes(j).point, d_planes(j).normal);
                double d2 = dot(p2 - d_planes(j).point, d_planes(j).normal);

                if (d1 * d2 < 0.0) {
                    // Segment crosses the plane — compute intersection.
                    double t = d1 / (d1 - d2);
                    Vec3 cross_pos = p1 + t * (p2 - p1);

                    int idx = Kokkos::atomic_fetch_add(&count(), 1);
                    if (idx < max_events) {
                        events(idx).seg_id   = s;
                        events(idx).plane_id = d_planes(j).id;
                        events(idx).cross_pos = cross_pos;
                        events(idx).normal   = d_planes(j).normal;
                    }
                }
            }
        });
        Kokkos::fence();

        // Read event count on host
        auto h_count = Kokkos::create_mirror_view(count);
        Kokkos::deep_copy(h_count, count);
        int nevents = h_count();
        if (nevents > max_events) nevents = max_events;

        // --- Phase 2: serial split_seg execution on host ---
        // get_serial_network() syncs device → serial automatically.
        SerialDisNet* network = system->get_serial_network();

        if (nevents > 0) {
            auto h_events = Kokkos::create_mirror_view(events);
            Kokkos::deep_copy(h_events, events);

            // Skip duplicate segments (one split per segment per step).
            std::vector<bool> split_done(network->number_of_segs(), false);

            for (int k = 0; k < nevents; k++) {
                int seg_id = h_events(k).seg_id;
                if (seg_id < 0 || seg_id >= (int)split_done.size()) continue;
                if (split_done[seg_id]) continue;
                split_done[seg_id] = true;

                Vec3 cross_pos = network->cell.pbc_fold(h_events(k).cross_pos);
                int nnew = network->split_seg(seg_id, cross_pos);

                network->nodes[nnew].constraint  = TWIN_SURFACE;
                network->nodes[nnew].twin_id     = h_events(k).plane_id;
                network->nodes[nnew].twin_normal = h_events(k).normal;
            }
        }

        // --- Phase 3: refresh / release existing TWIN_SURFACE nodes ---
        const double release_tol = 8.0;
        int Nnodes = network->number_of_nodes();

        for (int i = 0; i < Nnodes; i++) {
            if (network->nodes[i].constraint != TWIN_SURFACE) continue;
            int tid = network->nodes[i].twin_id;

            for (const PlanarObstacle& plane : system->planar_obstacles) {
                if (plane.id != tid) continue;

                double dist = dot(network->nodes[i].pos - plane.point, plane.normal);
                if (fabs(dist) > release_tol) {
                    // Far from plane — release constraint.
                    network->nodes[i].constraint  = UNCONSTRAINED;
                    network->nodes[i].twin_id     = -1;
                    network->nodes[i].twin_normal = Vec3(0.0);
                } else {
                    // Snap back onto plane (correct numerical drift).
                    network->nodes[i].pos = network->nodes[i].pos - dist * plane.normal;
                }
                break;
            }
        }

        network->update_ptr();
        system->net_mngr->set_network(network);
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

        // 3. Twin-boundary split_seg + velocity projection (parallel + serial).
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
