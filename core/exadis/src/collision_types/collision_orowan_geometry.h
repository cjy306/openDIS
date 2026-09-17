/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Isolated hard-sphere Orowan contact creation and release.
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_COLLISION_OROWAN_GEOMETRY_H
#define EXADIS_COLLISION_OROWAN_GEOMETRY_H

#include "collision_retroactive.h"
#include "orowan_geometry.h"

namespace ExaDiS {

class CollisionOrowanGeometry : public CollisionRetroactive {
private:
    double contact_tolerance = 1.0e-8;
    double drift_tolerance = 1.0e-5;
    double endpoint_tolerance = 1.0e-6;

    Vec3 exact_contact_position(const DisSeg& segment,
                                const OrowanGeometry::StaticContact& contact,
                                double radius,
                                double correction_ceiling) const
    {
        Vec3 glide_normal = segment.plane.normalized();
        if (glide_normal.norm2() <= 0.5)
            ExaDiS_fatal("Error: Orowan contact segment has no valid glide plane\n");

        double center_offset = dot(contact.center_image - contact.point, glide_normal);
        double circle_radius2 = radius * radius - center_offset * center_offset;
        if (circle_radius2 <= 0.0)
            ExaDiS_fatal("Error: Orowan glide plane does not intersect the sphere\n");
        Vec3 circle_center = contact.center_image - center_offset * glide_normal;
        Vec3 direction = contact.point - circle_center;
        double direction_norm = direction.norm();
        if (direction_norm <= 1.0e-30)
            ExaDiS_fatal("Error: undefined direction on Orowan contact circle\n");
        Vec3 position = circle_center + sqrt(circle_radius2) / direction_norm * direction;
        if ((position - contact.point).norm() > correction_ceiling)
            ExaDiS_fatal("Error: Orowan contact requires a nonlocal position correction\n");
        return position;
    }

    bool adjacent_segments_safe(SerialDisNet* net, int node,
                                const SphericalObstacle& sphere) const
    {
        for (int arm = 0; arm < net->conn[node].num; ++arm) {
            int segment = net->conn[node].seg[arm];
            int n1 = net->segs[segment].n1;
            int n2 = net->segs[segment].n2;
            auto contact = OrowanGeometry::segment_sphere_static(
                net->cell, net->nodes[n1].pos, net->nodes[n2].pos,
                sphere.center, sphere.radius, contact_tolerance);
            if (contact.penetrates) return false;
        }
        return true;
    }

    void release_safe_contacts(System* system, SerialDisNet* net)
    {
        for (int node = 0; node < net->number_of_nodes(); ++node) {
            DisNode& current = net->nodes[node];
            if (current.constraint != SPHERE_SURFACE) continue;
            int obstacle_id = current.sphere_id;
            if (obstacle_id < 0 || obstacle_id >= (int)system->obstacles.size())
                ExaDiS_fatal("Error: invalid sphere id on Orowan contact node\n");
            const SphericalObstacle& sphere = system->obstacles[obstacle_id];
            if (sphere.type != OBSTACLE_OROWAN)
                ExaDiS_fatal("Error: Orowan contact node refers to a non-Orowan obstacle\n");

            Vec3 center = net->cell.pbc_position(current.pos, sphere.center);
            Vec3 radial = current.pos - center;
            double distance = radial.norm();
            if (distance <= 1.0e-30 || distance < sphere.radius - contact_tolerance)
                continue;
            if (dot(current.v, radial) <= 0.0) continue;
            if (!adjacent_segments_safe(net, node, sphere)) continue;

            current.constraint = UNCONSTRAINED;
            current.sphere_id = -1;
            current.sphere_normal = Vec3(0.0);
        }
    }

    void correct_active_contacts(System* system, SerialDisNet* net)
    {
        for (int node = 0; node < net->number_of_nodes(); ++node) {
            DisNode& current = net->nodes[node];
            if (current.constraint != SPHERE_SURFACE) continue;
            int obstacle_id = current.sphere_id;
            if (obstacle_id < 0 || obstacle_id >= (int)system->obstacles.size())
                ExaDiS_fatal("Error: invalid sphere id during Orowan drift correction\n");
            if (net->conn[node].num == 0)
                ExaDiS_fatal("Error: isolated Orowan contact node\n");

            const SphericalObstacle& sphere = system->obstacles[obstacle_id];
            int segment = net->conn[node].seg[0];
            OrowanGeometry::StaticContact contact;
            contact.point = current.pos;
            contact.center_image = net->cell.pbc_position(current.pos, sphere.center);
            Vec3 corrected = exact_contact_position(
                net->segs[segment], contact, sphere.radius, drift_tolerance);
            Vec3 current_image = net->cell.pbc_position(corrected, current.pos);
            double correction = (corrected - current_image).norm();
            if (correction > drift_tolerance)
                ExaDiS_fatal(
                    "Error: Orowan contact drift exceeds correction ceiling "
                    "(node=%d obstacle=%d drift=%.17g)\n",
                    node, obstacle_id, correction);

            current.pos = net->cell.pbc_fold(corrected);
            Vec3 center = net->cell.pbc_position(current.pos, sphere.center);
            current.sphere_normal = (current.pos - center).normalized();
        }
    }

    void create_contacts(System* system, SerialDisNet* net)
    {
        int initial_segments = net->number_of_segs();
        for (int segment = 0; segment < initial_segments; ++segment) {
            int n1 = net->segs[segment].n1;
            int n2 = net->segs[segment].n2;
            if (net->nodes[n1].constraint == PINNED_NODE ||
                net->nodes[n1].constraint == CORNER_NODE ||
                net->nodes[n2].constraint == PINNED_NODE ||
                net->nodes[n2].constraint == CORNER_NODE)
                continue;

            for (int obstacle = 0; obstacle < (int)system->obstacles.size(); ++obstacle) {
                const SphericalObstacle& sphere = system->obstacles[obstacle];
                if (sphere.type != OBSTACLE_OROWAN) continue;
                if ((net->nodes[n1].constraint == SPHERE_SURFACE &&
                     net->nodes[n1].sphere_id == obstacle) ||
                    (net->nodes[n2].constraint == SPHERE_SURFACE &&
                     net->nodes[n2].sphere_id == obstacle))
                    continue;

                auto contact = OrowanGeometry::segment_sphere_static(
                    net->cell, net->nodes[n1].pos, net->nodes[n2].pos,
                    sphere.center, sphere.radius, contact_tolerance);
                if (contact.penetrates)
                    ExaDiS_fatal(
                        "Error: accepted Orowan segment penetrates sphere "
                        "(segment=%d obstacle=%d gap=%.17g)\n",
                        segment, obstacle, contact.gap);
                if (!contact.touches) continue;

                Vec3 position = exact_contact_position(
                    net->segs[segment], contact, sphere.radius,
                    10.0 * contact_tolerance);
                Vec3 endpoint1 = net->cell.pbc_position(position, net->nodes[n1].pos);
                Vec3 endpoint2 = net->cell.pbc_position(position, net->nodes[n2].pos);
                bool reuse_n1 = contact.s <= endpoint_tolerance &&
                    (endpoint1 - position).norm() <= 10.0 * contact_tolerance;
                bool reuse_n2 = contact.s >= 1.0 - endpoint_tolerance &&
                    (endpoint2 - position).norm() <= 10.0 * contact_tolerance;

                int contact_node;
                if (reuse_n1) {
                    contact_node = n1;
                } else if (reuse_n2) {
                    contact_node = n2;
                } else {
                    contact_node = net->split_seg(segment, net->cell.pbc_fold(position));
                }

                DisNode& node = net->nodes[contact_node];
                node.pos = net->cell.pbc_fold(position);
                Vec3 center = net->cell.pbc_position(node.pos, sphere.center);
                Vec3 radial = node.pos - center;
                double radial_norm = radial.norm();
                if (radial_norm <= 1.0e-30)
                    ExaDiS_fatal("Error: Orowan contact node lies at sphere center\n");
                node.constraint = SPHERE_SURFACE;
                node.sphere_id = obstacle;
                node.sphere_normal = (1.0 / radial_norm) * radial;
                break;
            }
        }
    }

    void audit_segments(System* system, SerialDisNet* net) const
    {
        for (int segment = 0; segment < net->number_of_segs(); ++segment) {
            int n1 = net->segs[segment].n1;
            int n2 = net->segs[segment].n2;
            for (int obstacle = 0; obstacle < (int)system->obstacles.size(); ++obstacle) {
                const SphericalObstacle& sphere = system->obstacles[obstacle];
                if (sphere.type != OBSTACLE_OROWAN) continue;
                auto contact = OrowanGeometry::segment_sphere_static(
                    net->cell, net->nodes[n1].pos, net->nodes[n2].pos,
                    sphere.center, sphere.radius, contact_tolerance);
                if (contact.penetrates)
                    ExaDiS_fatal(
                        "Error: Orowan segment audit failed "
                        "(segment=%d obstacle=%d gap=%.17g)\n",
                        segment, obstacle, contact.gap);
            }
        }
    }

public:
    CollisionOrowanGeometry(System* system) : CollisionRetroactive(system) {}

    void handle(System* system) override
    {
        CollisionRetroactive::handle(system);
        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].start();
        SerialDisNet* net = system->get_serial_network();
        release_safe_contacts(system, net);
        correct_active_contacts(system, net);
        create_contacts(system, net);
        audit_segments(system, net);
        system->get_device_network();
        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }

    const char* name() override { return "CollisionOrowanGeometry"; }
};

} // namespace ExaDiS

#endif
