/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    CollisionOrowan
 *    Implements Orowan bypass for spherical precipitates.
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

#include "collision_retroactive.h"

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
                    Vec3   normal = (dist > 1e-10) ? (d / dist)
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
                            (dist > 1e-10) ? (d / dist) : Vec3(0.0, 0.0, 1.0);
                    }
                }
            }
        }

        // Sync modified SerialDisNet back to device memory.
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

        // 2. Orowan sphere-surface enforcement.
        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].start();
        handle_orowan(system);
        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    const char* name() { return "CollisionOrowan"; }
};

} // namespace ExaDiS

#endif
