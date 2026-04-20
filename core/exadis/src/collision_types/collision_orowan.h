/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    CollisionOrowan
 *    Implements Orowan bypass for spherical precipitates and
 *    twin-boundary blocking for planar obstacles.
 *
 *    Each simulation step, after the standard retroactive collision:
 *      1. Any node found inside a sphere is pushed back to the sphere
 *         surface and marked SPHERE_SURFACE.
 *      2. Nodes already on a sphere surface have their outward normal
 *         refreshed so the mobility tangential projection stays correct.
 *      3. If a SPHERE_SURFACE node has moved clearly outside its sphere
 *         (e.g. after a topology merge / Orowan bypass completion) the
 *         constraint is released automatically.
 *
 *    The actual bypass loop is produced naturally: when the two dislocation
 *    arms that wrap around a sphere approach each other on the far side,
 *    the existing CollisionRetroactive and TopologyParallel modules detect
 *    the near-intersection and perform a node merge / annihilation, which
 *    leaves a closed dislocation ring around the sphere - the Orowan loop.
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
     *  handle_orowan
     *  Works on SerialDisNet (host), then syncs back to device.
     *---------------------------------------------------------------------*/
    void handle_orowan(System* system)
    {
        if (system->obstacles.empty()) return;

        SerialDisNet* net = system->get_serial_network();
        int Nnodes = net->number_of_nodes();

        for (int i = 0; i < Nnodes; i++) {
            Vec3& pos = net->nodes[i].pos;

            for (const SphericalObstacle& obs : system->obstacles) {

                Vec3   d     = pos - obs.center;
                double dist2 = d.norm2();
                double r2    = obs.radius * obs.radius;

                if (dist2 < r2) {
                    // Node is inside the sphere: push to surface and constrain.
                    double dist   = sqrt(dist2);
                    Vec3   normal = (dist > 1e-10) ? (1.0/dist) * d
                                                   : Vec3(0.0, 0.0, 1.0);
                    pos = obs.center + obs.radius * normal;

                    net->nodes[i].constraint    = SPHERE_SURFACE;
                    net->nodes[i].sphere_id     = obs.id;
                    net->nodes[i].sphere_normal = normal;

                } else if (net->nodes[i].constraint == SPHERE_SURFACE &&
                           net->nodes[i].sphere_id  == obs.id) {
                    // Node is already on this sphere surface.
                    double dist = sqrt(dist2);

                    if (dist > obs.radius * 1.05) {
                        // Moved clearly away (topology handled it): release.
                        net->nodes[i].constraint    = UNCONSTRAINED;
                        net->nodes[i].sphere_id     = -1;
                        net->nodes[i].sphere_normal = Vec3(0.0);
                    } else {
                        // Still on surface: refresh outward normal.
                        net->nodes[i].sphere_normal =
                            (dist > 1e-10) ? (1.0/dist) * d : Vec3(0.0, 0.0, 1.0);
                    }
                }
            }
        }

        // Sync modified SerialDisNet back to device memory.
        net->update_ptr();
        system->net_mngr->set_network(net);
    }

    /*-----------------------------------------------------------------------
     *  handle_twin
     *  Enforce twin-boundary plane constraints.
     *  Any node that crosses a twin plane is pushed back onto the plane
     *  and marked TWIN_SURFACE.  Nodes already constrained have their
     *  normal refreshed or are released if they moved far away.
     *---------------------------------------------------------------------*/
    void handle_twin(System* system)
    {
        if (system->planar_obstacles.empty()) return;

        SerialDisNet* net = system->get_serial_network();
        int Nnodes = net->number_of_nodes();

        // Thickness tolerance: nodes within this distance are "on" the plane
        double tol = 1.0; // 1 Burgers vector

        for (int i = 0; i < Nnodes; i++) {
            Vec3& pos = net->nodes[i].pos;

            for (const PlanarObstacle& plane : system->planar_obstacles) {

                // Signed distance from node to plane
                Vec3   d    = pos - plane.point;
                double dist = dot(d, plane.normal);

                if (fabs(dist) < tol) {
                    // Node is within tolerance of the twin plane: snap to plane
                    pos = pos - dist * plane.normal;

                    net->nodes[i].constraint  = TWIN_SURFACE;
                    net->nodes[i].twin_id     = plane.id;
                    net->nodes[i].twin_normal = plane.normal;

                } else if (net->nodes[i].constraint == TWIN_SURFACE &&
                           net->nodes[i].twin_id    == plane.id) {
                    // Node was on this twin plane
                    if (fabs(dist) > tol * 3.0) {
                        // Moved clearly away: release constraint
                        net->nodes[i].constraint  = UNCONSTRAINED;
                        net->nodes[i].twin_id     = -1;
                        net->nodes[i].twin_normal = Vec3(0.0);
                    } else {
                        // Still near surface: snap back
                        pos = pos - dist * plane.normal;
                    }
                }
            }
        }

        net->update_ptr();
        system->net_mngr->set_network(net);
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

        // 2. Orowan sphere-surface enforcement.
        handle_orowan(system);

        // 3. Twin-boundary plane enforcement.
        handle_twin(system);

        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
