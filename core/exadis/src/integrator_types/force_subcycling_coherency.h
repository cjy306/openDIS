/*---------------------------------------------------------------------------
 *
 *  Subcycling force with a periodic, precomputed coherency-stress field.
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_SUBCYCLING_COHERENCY_H
#define EXADIS_FORCE_SUBCYCLING_COHERENCY_H

#include "integrator_subcycling.h"

namespace ExaDiS {

class ForceSubcyclingCoherency : public ForceSubcycling {
public:
    struct Params {
        ForceSubcycling::Params subcycling;
        int Nx, Ny, Nz;
        std::vector<double> stress;

        Params(ForceSubcycling::Params _subcycling, int _Nx, int _Ny, int _Nz,
               const std::vector<double>& _stress) :
            subcycling(_subcycling), Nx(_Nx), Ny(_Ny), Nz(_Nz), stress(_stress) {}
    };

    int Nx, Ny, Nz;
    Kokkos::View<double*> stress;
    Kokkos::View<double*>::HostMirror h_stress;

    ForceSubcyclingCoherency(System* system, Params params) :
        ForceSubcycling(system, params.subcycling),
        Nx(params.Nx), Ny(params.Ny), Nz(params.Nz),
        stress("coherency_stress", params.stress.size())
    {
        h_stress = Kokkos::create_mirror_view(stress);
        for (size_t i = 0; i < params.stress.size(); ++i)
            h_stress(i) = params.stress[i];
        Kokkos::deep_copy(stress, h_stress);
    }

    template<class ViewType>
    KOKKOS_INLINE_FUNCTION
    Mat33 interpolate_stress(const Cell& cell, const Vec3& position,
                             const ViewType& field) const
    {
        Vec3 scaled = cell.scaled_position(position);
        scaled.x -= floor(scaled.x);
        scaled.y -= floor(scaled.y);
        scaled.z -= floor(scaled.z);

        double grid_x = scaled.x * Nx;
        double grid_y = scaled.y * Ny;
        double grid_z = scaled.z * Nz;
        int i0 = (int)floor(grid_x);
        int j0 = (int)floor(grid_y);
        int k0 = (int)floor(grid_z);
        int i1 = (i0 + 1 == Nx) ? 0 : i0 + 1;
        int j1 = (j0 + 1 == Ny) ? 0 : j0 + 1;
        int k1 = (k0 + 1 == Nz) ? 0 : k0 + 1;
        double tx = grid_x - i0;
        double ty = grid_y - j0;
        double tz = grid_z - k0;

        double value[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int di = 0; di < 2; ++di) {
            int i = di ? i1 : i0;
            double wi = di ? tx : 1.0 - tx;
            for (int dj = 0; dj < 2; ++dj) {
                int j = dj ? j1 : j0;
                double wj = dj ? ty : 1.0 - ty;
                for (int dk = 0; dk < 2; ++dk) {
                    int k = dk ? k1 : k0;
                    double wk = dk ? tz : 1.0 - tz;
                    int offset = ((i * Ny + j) * Nz + k) * 6;
                    double weight = wi * wj * wk;
                    for (int component = 0; component < 6; ++component)
                        value[component] += weight * field(offset + component);
                }
            }
        }

        // Fixed component order: xx, yy, zz, yz, xz, xy.
        return Mat33().set(
            value[0], value[5], value[4],
            value[5], value[1], value[3],
            value[4], value[3], value[2]
        );
    }

    template<class N, class ViewType>
    KOKKOS_INLINE_FUNCTION
    SegForce segment_coherency_force(N* net, const int& segment_index,
                                     const ViewType& field) const
    {
        auto nodes = net->get_nodes();
        auto segments = net->get_segs();
        int n1 = segments[segment_index].n1;
        int n2 = segments[segment_index].n2;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = net->cell.pbc_position(r1, nodes[n2].pos);
        Vec3 midpoint = 0.5 * (r1 + r2);
        Mat33 sigma = interpolate_stress(net->cell, midpoint, field);

        // PHYS-APPROX: one midpoint quadrature point per segment; ceiling =
        // unresolved stress variation within long segments; upgrade = two-point
        // or adaptive Gauss integration with distinct endpoint shape functions.
        Vec3 force = pk_force(segments[segment_index].burg, r1, r2, sigma);
        return SegForce(force, force);
    }

    struct AddCoherencyForce {
        DeviceDisNet* net;
        ForceSubcyclingCoherency* force;

        AddCoherencyForce(DeviceDisNet* _net, ForceSubcyclingCoherency* _force) :
            net(_net), force(_force) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const int& segment_index) const
        {
            auto nodes = net->get_nodes();
            auto segments = net->get_segs();
            SegForce segment_force = force->segment_coherency_force(
                net, segment_index, force->stress
            );
            Kokkos::atomic_add(&nodes[segments[segment_index].n1].f, segment_force.f1);
            Kokkos::atomic_add(&nodes[segments[segment_index].n2].f, segment_force.f2);
        }
    };

    void compute(System* system, bool zero=true) override
    {
        ForceSubcycling::compute(system, zero);
        if (group != 0)
            return;

        DeviceDisNet* net = system->get_device_network();
        Kokkos::parallel_for(
            "AddCoherencyStressForce", net->Nsegs_local,
            AddCoherencyForce(net, this)
        );
        Kokkos::fence();
    }

    Vec3 node_force(System* system, const int& node_index) override
    {
        Vec3 force = ForceSubcycling::node_force(system, node_index);
        SerialDisNet* net = system->get_serial_network();
        auto connections = net->get_conn();
        for (int j = 0; j < connections[node_index].num; ++j) {
            int segment_index = connections[node_index].seg[j];
            SegForce segment_force = segment_coherency_force(
                net, segment_index, h_stress
            );
            force += (connections[node_index].order[j] == 1) ?
                segment_force.f1 : segment_force.f2;
        }
        return force;
    }

    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_force(System* system, N* net, const int& node_index,
                    const team_handle& team)
    {
        Vec3 force = ForceSubcycling::node_force(system, net, node_index, team);
        auto connections = net->get_conn();
        Vec3 coherency_force(0.0);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, connections[node_index].num),
            [&] (const int& j, Vec3& sum) {
                int segment_index = connections[node_index].seg[j];
                SegForce segment_force = segment_coherency_force(
                    net, segment_index, stress
                );
                sum += (connections[node_index].order[j] == 1) ?
                    segment_force.f1 : segment_force.f2;
            }, coherency_force
        );
        team.team_barrier();
        return force + coherency_force;
    }

    // PHYS-APPROX: coherency stress only; ceiling = no local Cr-dependent drag;
    // upgrade = a separate MobilityCrDependent model validated independently.
    const char* name() override { return "ForceSubcyclingCoherency"; }
};

namespace ForceType {
    typedef ForceSubcyclingCoherency SUBCYCLING_COHERENCY_MODEL;
}

} // namespace ExaDiS

#endif
