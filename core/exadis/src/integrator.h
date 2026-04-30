/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INTEGRATOR_H
#define EXADIS_INTEGRATOR_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Integrator
 *
 *-------------------------------------------------------------------------*/
class Integrator {
public:
    double nextdt;
public:
    Integrator() {}
    Integrator(System* system) {}
    virtual void integrate(System* system) {}
    virtual KOKKOS_FUNCTION ~Integrator() {}
    virtual const char* name() { return "IntegratorNone"; }
    
    // Restart
    virtual void write_restart(FILE* fp) {
        fprintf(fp, "nextdt %.17g\n", nextdt);
    }
    virtual void read_restart(FILE* fp) {
        fscanf(fp, "nextdt %lf\n", &nextdt);
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorEuler
 *
 *-------------------------------------------------------------------------*/
class IntegratorEuler : public Integrator {    
public:
    IntegratorEuler(System* system) {
        nextdt = system->params.nextdt;
    }
    
    void integrate(System* system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].start();

        double dt = nextdt;

        DeviceDisNet* net = system->get_device_network();

        Kokkos::resize(system->xold, net->Nnodes_local);

        // Copy twin planes to device for position clamping
        int n_twin = (int)system->planar_obstacles.size();
        Kokkos::View<PlanarObstacle*, T_memory_space> d_planes("d_planes_euler", n_twin);
        if (n_twin > 0) {
            auto h_planes = Kokkos::create_mirror_view(d_planes);
            for (int j = 0; j < n_twin; j++)
                h_planes(j) = system->planar_obstacles[j];
            Kokkos::deep_copy(d_planes, h_planes);
        }

        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int i) {
            auto nodes = net->get_nodes();
            auto cell = net->cell;

            system->xold(i) = nodes[i].pos;
            Vec3 rnew = nodes[i].pos + dt*nodes[i].v;
            nodes[i].pos = cell.pbc_fold(rnew);

            // Clamp TWIN_SURFACE nodes back onto their twin plane
            if (nodes[i].constraint == TWIN_SURFACE && n_twin > 0) {
                int tid = nodes[i].twin_id;
                for (int p = 0; p < n_twin; p++) {
                    if (d_planes(p).id == tid) {
                        double d = dot(nodes[i].pos - d_planes(p).point,
                                       d_planes(p).normal);
                        nodes[i].pos = nodes[i].pos - d * d_planes(p).normal;
                        break;
                    }
                }
            }
        });

        system->realdt = dt;

        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].stop();
    }
    
    const char* name() { return "IntegratorEuler"; }
};

} // namespace ExaDiS


#include "integrator_trapezoid.h"
#include "integrator_rkf.h"
#include "integrator_multi.h"
#include "integrator_subcycling.h"

#endif
