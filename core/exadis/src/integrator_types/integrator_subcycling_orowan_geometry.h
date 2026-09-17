/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Transactional Subcycling step guard for hard-sphere Orowan geometry.
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INTEGRATOR_SUBCYCLING_OROWAN_GEOMETRY_H
#define EXADIS_INTEGRATOR_SUBCYCLING_OROWAN_GEOMETRY_H

#include "integrator_subcycling.h"
#include "orowan_geometry.h"

namespace ExaDiS {

class IntegratorSubcyclingOrowanGeometry : public IntegratorSubcycling {
public:
    typedef IntegratorSubcycling::Params Params;
    typedef IntegratorSubcycling::AdaptiveState AdaptiveState;

private:
    int max_geometry_retries = 24;
    double contact_tolerance = 1.0e-8;
    double event_tolerance = 1.0e-6;

    struct SavePositions {
        DeviceDisNet* net;
        T_x positions;
        SavePositions(DeviceDisNet* _net, T_x _positions) : net(_net), positions(_positions) {}
        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const { positions(i) = net->get_nodes()[i].pos; }
    };

    struct RestorePositions {
        DeviceDisNet* net;
        T_x positions;
        RestorePositions(DeviceDisNet* _net, T_x _positions) : net(_net), positions(_positions) {}
        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const { net->get_nodes()[i].pos = positions(i); }
    };

    bool has_hard_spheres(System* system) const
    {
        for (const auto& obstacle : system->obstacles)
            if (obstacle.type == OBSTACLE_OROWAN) return true;
        return false;
    }

    bool geometry_requires_retry(System* system, double& earliest_alpha,
                                 int& hit_segment, int& hit_obstacle)
    {
        SerialDisNet* net = system->get_serial_network();
#if EXADIS_FULL_UNIFIED_MEMORY
        T_x& old_positions = system->xold;
#else
        T_x::HostMirror old_positions = Kokkos::create_mirror_view(system->xold);
        Kokkos::deep_copy(old_positions, system->xold);
#endif
        earliest_alpha = 1.0;
        hit_segment = -1;
        hit_obstacle = -1;
        bool retry = false;

        for (int segment = 0; segment < net->number_of_segs(); ++segment) {
            int n1 = net->segs[segment].n1;
            int n2 = net->segs[segment].n2;
            Vec3 old1 = old_positions(n1);
            Vec3 old2 = old_positions(n2);
            Vec3 new1 = net->nodes[n1].pos;
            Vec3 new2 = net->nodes[n2].pos;

            for (int obstacle = 0; obstacle < (int)system->obstacles.size(); ++obstacle) {
                const SphericalObstacle& sphere = system->obstacles[obstacle];
                if (sphere.type != OBSTACLE_OROWAN) continue;

                OrowanGeometry::StaticContact final_contact =
                    OrowanGeometry::segment_sphere_static(
                        net->cell, new1, new2, sphere.center,
                        sphere.radius, contact_tolerance);
                OrowanGeometry::SweptContact swept =
                    OrowanGeometry::moving_segment_sphere_first_contact(
                        net->cell, old1, old2, new1, new2, sphere.center,
                        sphere.radius, contact_tolerance);

                bool new_event_before_end = swept.hit &&
                    swept.alpha < 1.0 - event_tolerance;
                if (final_contact.penetrates || new_event_before_end) {
                    double alpha = swept.hit ? swept.alpha : 0.5;
                    if (!retry || alpha < earliest_alpha) {
                        retry = true;
                        earliest_alpha = alpha;
                        hit_segment = segment;
                        hit_obstacle = obstacle;
                    }
                }
            }
        }
        return retry;
    }

public:
    IntegratorSubcyclingOrowanGeometry(System* system, Force* force, Mobility* mobility,
                                       Params params=Params()) :
        IntegratorSubcycling(system, force, mobility, params) {}

    void integrate(System* system) override
    {
        if (!has_hard_spheres(system)) {
            IntegratorSubcycling::integrate(system);
            return;
        }

        DeviceDisNet* device_network = system->get_device_network();
        T_x step_start("orowan_geometry_step_start", device_network->Nnodes_local);
        Kokkos::parallel_for(device_network->Nnodes_local,
                             SavePositions(device_network, step_start));
        Kokkos::fence();
        double step_start_realdt = system->realdt;

        for (int retry_count = 0; retry_count <= max_geometry_retries; ++retry_count) {
            AdaptiveState adaptive_start;
            save_adaptive_state(adaptive_start);
            IntegratorSubcycling::integrate(system);

            double earliest_alpha = 1.0;
            int hit_segment = -1;
            int hit_obstacle = -1;
            if (!geometry_requires_retry(system, earliest_alpha,
                                         hit_segment, hit_obstacle))
                return;

            if (retry_count == max_geometry_retries) {
                ExaDiS_fatal(
                    "Error: Orowan geometry failed to reach contact after %d retries "
                    "(segment=%d obstacle=%d alpha=%.17g dt=%.17g)\n",
                    max_geometry_retries, hit_segment, hit_obstacle,
                    earliest_alpha, system->realdt);
            }

            double attempted_dt = system->realdt;
            double alpha_for_retry = fmax(0.05, fmin(0.999999, earliest_alpha));
            double retry_dt = attempted_dt * alpha_for_retry;
            if (retry_dt <= 1.0e-20)
                ExaDiS_fatal("Error: Orowan geometry retry dt underflow\n");

            device_network = system->get_device_network();
            Kokkos::parallel_for(device_network->Nnodes_local,
                                 RestorePositions(device_network, step_start));
            Kokkos::fence();
            restore_adaptive_state(adaptive_start);
            limit_global_nextdt(retry_dt);
            system->realdt = step_start_realdt;
        }
    }

    const char* name() override { return "IntegratorSubcyclingOrowanGeometry"; }
};

} // namespace ExaDiS

#endif
