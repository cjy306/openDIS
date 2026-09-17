/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Contact-preserving remesh path for hard-sphere Orowan geometry.
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_REMESH_OROWAN_GEOMETRY_H
#define EXADIS_REMESH_OROWAN_GEOMETRY_H

#include "orowan_geometry.h"

namespace ExaDiS {

class RemeshOrowanGeometry : public RemeshSerial {
private:
    bool has_active_contact(SerialDisNet* network) const
    {
        for (int i = 0; i < network->number_of_nodes(); ++i)
            if (network->nodes[i].constraint == SPHERE_SURFACE) return true;
        return false;
    }

    void refine_without_contact_coarsening(System* system, SerialDisNet* network)
    {
        double maxseg = system->params.maxseg;
        int initial_segments = network->number_of_segs();
        for (int segment = 0; segment < initial_segments; ++segment) {
            int n1 = network->segs[segment].n1;
            int n2 = network->segs[segment].n2;
            Vec3 p1 = network->nodes[n1].pos;
            Vec3 p2 = network->cell.pbc_position(p1, network->nodes[n2].pos);
            if ((p2 - p1).norm() <= maxseg) continue;
            if (network->nodes[n1].constraint == PINNED_NODE &&
                network->nodes[n2].constraint == PINNED_NODE) continue;
            network->split_seg(segment, network->cell.pbc_fold(0.5 * (p1 + p2)));
        }
    }

    void audit_hard_spheres(System* system, SerialDisNet* network) const
    {
        const double tolerance = 1.0e-8;
        for (int segment = 0; segment < network->number_of_segs(); ++segment) {
            int n1 = network->segs[segment].n1;
            int n2 = network->segs[segment].n2;
            for (int obstacle = 0; obstacle < (int)system->obstacles.size(); ++obstacle) {
                const SphericalObstacle& sphere = system->obstacles[obstacle];
                if (sphere.type != OBSTACLE_OROWAN) continue;
                auto contact = OrowanGeometry::segment_sphere_static(
                    network->cell, network->nodes[n1].pos, network->nodes[n2].pos,
                    sphere.center, sphere.radius, tolerance);
                if (contact.penetrates)
                    ExaDiS_fatal(
                        "Error: topology/remesh created an Orowan sphere penetration "
                        "(segment=%d obstacle=%d gap=%.17g)\n",
                        segment, obstacle, contact.gap);
            }
        }
    }

public:
    RemeshOrowanGeometry(System* system, Params params=Params()) :
        RemeshSerial(system, params) {}

    void remesh(System* system) override
    {
        SerialDisNet* network = system->get_serial_network();
        if (!has_active_contact(network)) {
            RemeshSerial::remesh(system);
            audit_hard_spheres(system, network);
            return;
        }

        Kokkos::fence();
        system->timer[system->TIMER_REMESH].start();
        // PHYS-APPROX: coarsening is suspended globally while a sphere contact is active;
        // ceiling = temporary excess discretization away from the obstacle;
        // upgrade = obstacle-local merge guards that preserve every exterior replacement chord.
        refine_without_contact_coarsening(system, network);
        audit_hard_spheres(system, network);
        Kokkos::fence();
        system->timer[system->TIMER_REMESH].stop();
    }

    const char* name() override { return "RemeshOrowanGeometry"; }
};

} // namespace ExaDiS

#endif
