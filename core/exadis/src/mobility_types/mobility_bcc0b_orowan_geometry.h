/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    BCC_0B mobility with isolated hard-sphere Orowan kinematics.
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_BCC0B_OROWAN_GEOMETRY_H
#define EXADIS_MOBILITY_BCC0B_OROWAN_GEOMETRY_H

#include "mobility_bcc0b.h"
#include "orowan_geometry.h"

namespace ExaDiS {

class MobilityBCC0bOrowanGeometry : public Mobility {
public:
    typedef MobilityBCC0b::Params Params;

private:
    MobilityBCC0b* base;

    template<class N>
    KOKKOS_INLINE_FUNCTION
    static Vec3 constrain_contact_velocity(N* net, const int i, const Vec3& free_velocity,
                                           const SphericalObstacle& obstacle,
                                           double tangent_tol, double release_tol)
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto conn = net->get_conn();
        Vec3 center = net->cell.pbc_position(nodes[i].pos, obstacle.center);
        Vec3 radial = nodes[i].pos - center;
        double radial_norm = radial.norm();
        if (radial_norm <= 1.0e-30) return Vec3(0.0);

        Vec3 sphere_normal = (1.0 / radial_norm) * radial;
        nodes[i].sphere_normal = sphere_normal;
        if (dot(free_velocity, sphere_normal) > release_tol)
            return free_velocity;

        Vec3 plane1(0.0), plane2(0.0);
        int number_of_planes = 0;
        for (int arm = 0; arm < conn[i].num; ++arm) {
            Vec3 plane = segs[conn[i].seg[arm]].plane;
            double plane_norm = plane.norm();
            if (plane_norm <= tangent_tol) continue;
            plane = (1.0 / plane_norm) * plane;
            if (number_of_planes == 0) {
                plane1 = plane;
                number_of_planes = 1;
            } else if (fabs(dot(plane1, plane)) < 1.0 - 1.0e-6) {
                if (number_of_planes == 1) {
                    plane2 = plane;
                    number_of_planes = 2;
                } else if (fabs(dot(plane2, plane)) < 1.0 - 1.0e-6) {
                    return Vec3(0.0);
                }
            }
        }

        // PHYS-APPROX: frictionless kinematic hard-sphere contact;
        // ceiling = no interface strength, image stress, or cutting transition;
        // upgrade = a separately calibrated finite-strength/interface model.
        if (number_of_planes == 1) {
            bool mobile = false;
            return OrowanGeometry::common_tangent_velocity(
                free_velocity, plane1, sphere_normal, tangent_tol, &mobile);
        }
        if (number_of_planes == 2) {
            Vec3 junction = cross(plane1, plane2);
            double junction_norm = junction.norm();
            if (junction_norm <= tangent_tol) return Vec3(0.0);
            junction = (1.0 / junction_norm) * junction;
            if (fabs(dot(junction, sphere_normal)) > 1.0e-6)
                return Vec3(0.0);
            return dot(free_velocity, junction) * junction;
        }
        return Vec3(0.0);
    }

public:
    struct ContactMobility {
        MobilityBCC0b* base;
        Kokkos::View<SphericalObstacle*, T_memory_space> obstacles;
        int Nobstacles = 0;
        double tangent_tolerance = 1.0e-10;
        double release_velocity_tolerance = 1.0e-12;

        ContactMobility(MobilityBCC0b* _base) : base(_base) {}

        template<class N>
        KOKKOS_INLINE_FUNCTION
        Vec3 node_velocity(System* system, N* net, const int i, const Vec3& force) const
        {
            auto nodes = net->get_nodes();
            Vec3 velocity = base->node_velocity(system, net, i, force);
            if (nodes[i].constraint != SPHERE_SURFACE) return velocity;

            int obstacle_id = nodes[i].sphere_id;
            if (obstacle_id < 0 || obstacle_id >= Nobstacles ||
                obstacles(obstacle_id).type != OBSTACLE_OROWAN)
                return Vec3(0.0);

            return MobilityBCC0bOrowanGeometry::constrain_contact_velocity(
                net, i, velocity, obstacles(obstacle_id),
                tangent_tolerance, release_velocity_tolerance);
        }
    };

    typedef ContactMobility Mob;
    Mob* mob;

    MobilityBCC0bOrowanGeometry(System* system, Params params)
    {
        base = exadis_new<MobilityBCC0b>(system, params);
        mob = exadis_new<Mob>(base);
        non_linear = base->non_linear;
        refresh_obstacles(system);
    }

    struct NodeMobility {
        System* system;
        Mob* mob;
        DeviceDisNet* net;

        NodeMobility(System* _system, Mob* _mob, DeviceDisNet* _net) :
            system(_system), mob(_mob), net(_net) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const
        {
            auto nodes = net->get_nodes();
            nodes[i].v = mob->node_velocity(system, net, i, nodes[i].f);
        }
    };

    void refresh_obstacles(System* system)
    {
        mob->Nobstacles = (int)system->obstacles.size();
        Kokkos::resize(mob->obstacles, mob->Nobstacles);
        if (mob->Nobstacles == 0) return;
        auto host = Kokkos::create_mirror_view(mob->obstacles);
        for (int i = 0; i < mob->Nobstacles; ++i) host(i) = system->obstacles[i];
        Kokkos::deep_copy(mob->obstacles, host);
    }

    void compute(System* system) override
    {
        Kokkos::fence();
        system->timer[system->TIMER_MOBILITY].start();
        refresh_obstacles(system);
        DeviceDisNet* net = system->get_device_network();
        using policy = Kokkos::RangePolicy<Kokkos::LaunchBounds<32,1>>;
        Kokkos::parallel_for(
            "MobilityBCC0bOrowanGeometry", policy(0, net->Nnodes_local),
            NodeMobility(system, mob, net));
        Kokkos::fence();
        system->timer[system->TIMER_MOBILITY].stop();
    }

    Vec3 node_velocity(System* system, const int& i, const Vec3& force) override
    {
        if (mob->Nobstacles != (int)system->obstacles.size())
            refresh_obstacles(system);
        SerialDisNet* net = system->get_serial_network();
        return mob->node_velocity(system, net, i, force);
    }

    ~MobilityBCC0bOrowanGeometry() override {
        exadis_delete(mob);
        exadis_delete(base);
    }

    const char* name() override { return "MobilityBCC0bOrowanGeometry"; }
};

namespace MobilityType {
    typedef MobilityBCC0bOrowanGeometry BCC_0B_OROWAN_GEOMETRY;
}

} // namespace ExaDiS

#endif
