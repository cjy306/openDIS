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
    Kokkos::View<SphericalObstacle*, T_memory_space> device_obstacles;
    int Nobstacles = 0;
    double tangent_tolerance = 1.0e-10;
    double release_velocity_tolerance = 1.0e-12;

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
    MobilityBCC0bOrowanGeometry(System* system, Params params)
    {
        base = exadis_new<MobilityBCC0b>(system, params);
        non_linear = base->non_linear;
    }

    struct NodeMobility {
        System* system;
        MobilityBCC0b* base;
        DeviceDisNet* net;
        Kokkos::View<SphericalObstacle*, T_memory_space> obstacles;
        int Nobstacles;
        double tangent_tolerance;
        double release_velocity_tolerance;

        NodeMobility(System* _system, MobilityBCC0b* _base, DeviceDisNet* _net,
                     Kokkos::View<SphericalObstacle*, T_memory_space> _obstacles,
                     int _Nobstacles, double _tangent_tolerance,
                     double _release_velocity_tolerance) :
            system(_system), base(_base), net(_net), obstacles(_obstacles),
            Nobstacles(_Nobstacles), tangent_tolerance(_tangent_tolerance),
            release_velocity_tolerance(_release_velocity_tolerance) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const
        {
            auto nodes = net->get_nodes();
            Vec3 velocity = base->node_velocity(system, net, i, nodes[i].f);
            if (nodes[i].constraint == SPHERE_SURFACE) {
                int obstacle_id = nodes[i].sphere_id;
                if (obstacle_id < 0 || obstacle_id >= Nobstacles ||
                    obstacles(obstacle_id).type != OBSTACLE_OROWAN) {
                    nodes[i].v = Vec3(0.0);
                    return;
                }
                velocity = MobilityBCC0bOrowanGeometry::constrain_contact_velocity(
                    net, i, velocity, obstacles(obstacle_id),
                    tangent_tolerance, release_velocity_tolerance);
            }
            nodes[i].v = velocity;
        }
    };

    void refresh_obstacles(System* system)
    {
        Nobstacles = (int)system->obstacles.size();
        Kokkos::resize(device_obstacles, Nobstacles);
        if (Nobstacles == 0) return;
        auto host = Kokkos::create_mirror_view(device_obstacles);
        for (int i = 0; i < Nobstacles; ++i) host(i) = system->obstacles[i];
        Kokkos::deep_copy(device_obstacles, host);
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
            NodeMobility(system, base, net, device_obstacles, Nobstacles,
                         tangent_tolerance, release_velocity_tolerance));
        Kokkos::fence();
        system->timer[system->TIMER_MOBILITY].stop();
    }

    Vec3 node_velocity(System* system, const int& i, const Vec3& force) override
    {
        SerialDisNet* net = system->get_serial_network();
        Vec3 velocity = base->node_velocity(system, net, i, force);
        if (net->nodes[i].constraint != SPHERE_SURFACE) return velocity;
        int obstacle_id = net->nodes[i].sphere_id;
        if (obstacle_id < 0 || obstacle_id >= (int)system->obstacles.size() ||
            system->obstacles[obstacle_id].type != OBSTACLE_OROWAN)
            return Vec3(0.0);
        return constrain_contact_velocity(
            net, i, velocity, system->obstacles[obstacle_id],
            tangent_tolerance, release_velocity_tolerance);
    }

    ~MobilityBCC0bOrowanGeometry() override { exadis_delete(base); }

    const char* name() override { return "MobilityBCC0bOrowanGeometry"; }
};

namespace MobilityType {
    typedef MobilityBCC0bOrowanGeometry BCC_0B_OROWAN_GEOMETRY;
}

} // namespace ExaDiS

#endif
