/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Stateless geometry used by the isolated hard-sphere Orowan modules.
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_OROWAN_GEOMETRY_H
#define EXADIS_OROWAN_GEOMETRY_H

#include "network.h"

namespace ExaDiS {
namespace OrowanGeometry {

struct StaticContact {
    bool touches;
    bool penetrates;
    double gap;
    double s;
    Vec3 point;
    Vec3 normal;
    Vec3 center_image;

    KOKKOS_INLINE_FUNCTION
    StaticContact() : touches(false), penetrates(false), gap(0.0), s(0.0),
                      point(0.0), normal(0.0), center_image(0.0) {}
};

struct SweptContact {
    bool hit;
    double alpha;
    StaticContact contact;

    KOKKOS_INLINE_FUNCTION
    SweptContact() : hit(false), alpha(1.0), contact() {}
};

KOKKOS_INLINE_FUNCTION
StaticContact segment_sphere_static(const Cell& cell,
                                    const Vec3& p1,
                                    const Vec3& p2_raw,
                                    const Vec3& center_raw,
                                    double radius,
                                    double tolerance)
{
    StaticContact result;
    Vec3 p2 = cell.pbc_position(p1, p2_raw);
    Vec3 midpoint = 0.5 * (p1 + p2);
    Vec3 center = cell.pbc_position(midpoint, center_raw);
    Vec3 d = p2 - p1;
    double d2 = d.norm2();

    result.s = 0.0;
    if (d2 > 1.0e-30) {
        result.s = dot(center - p1, d) / d2;
        result.s = fmax(0.0, fmin(1.0, result.s));
    }

    result.point = p1 + result.s * d;
    result.center_image = center;
    Vec3 radial = result.point - center;
    double distance = radial.norm();
    result.gap = distance - radius;
    result.touches = (result.gap <= tolerance);
    result.penetrates = (result.gap < -tolerance);
    result.normal = (distance > 1.0e-30) ? (1.0 / distance) * radial : Vec3(0.0);
    return result;
}

KOKKOS_INLINE_FUNCTION
StaticContact segment_sphere_static_unwrapped(const Vec3& p1,
                                              const Vec3& p2,
                                              const Vec3& center,
                                              double radius,
                                              double tolerance)
{
    StaticContact result;
    Vec3 d = p2 - p1;
    double d2 = d.norm2();
    result.s = 0.0;
    if (d2 > 1.0e-30) {
        result.s = dot(center - p1, d) / d2;
        result.s = fmax(0.0, fmin(1.0, result.s));
    }
    result.point = p1 + result.s * d;
    result.center_image = center;
    Vec3 radial = result.point - center;
    double distance = radial.norm();
    result.gap = distance - radius;
    result.touches = (result.gap <= tolerance);
    result.penetrates = (result.gap < -tolerance);
    result.normal = (distance > 1.0e-30) ? (1.0 / distance) * radial : Vec3(0.0);
    return result;
}

KOKKOS_INLINE_FUNCTION
StaticContact swept_configuration(const Vec3& p10,
                                  const Vec3& p20,
                                  const Vec3& dp1,
                                  const Vec3& dp2,
                                  const Vec3& center,
                                  double radius,
                                  double tolerance,
                                  double alpha)
{
    return segment_sphere_static_unwrapped(p10 + alpha * dp1,
                                           p20 + alpha * dp2,
                                           center, radius, tolerance);
}

KOKKOS_INLINE_FUNCTION
SweptContact moving_segment_sphere_first_contact(const Cell& cell,
                                                 const Vec3& p10,
                                                 const Vec3& p20_raw,
                                                 const Vec3& p11_raw,
                                                 const Vec3& p21_raw,
                                                 const Vec3& center_raw,
                                                 double radius,
                                                 double tolerance)
{
    SweptContact result;
    Vec3 p20 = cell.pbc_position(p10, p20_raw);
    Vec3 p11 = cell.pbc_position(p10, p11_raw);
    Vec3 p21 = cell.pbc_position(p20, p21_raw);
    Vec3 center = cell.pbc_position(0.5 * (p10 + p20), center_raw);
    Vec3 dp1 = p11 - p10;
    Vec3 dp2 = p21 - p20;
    double speed_bound = fmax(dp1.norm(), dp2.norm());

    StaticContact start = segment_sphere_static_unwrapped(
        p10, p20, center, radius, tolerance);
    if (start.penetrates) {
        result.hit = true;
        result.alpha = 0.0;
        result.contact = start;
        return result;
    }
    if (speed_bound <= 1.0e-30) return result;

    // A legal alpha=0 tangency moving away must not be reported forever.
    double alpha = 0.0;
    if (start.touches) {
        double probe_alpha = fmin(1.0, fmax(1.0e-7, 4.0 * tolerance / speed_bound));
        StaticContact probe = swept_configuration(
            p10, p20, dp1, dp2, center, radius, tolerance, probe_alpha);
        if (probe.penetrates || probe.gap < start.gap - tolerance) {
            result.hit = true;
            result.alpha = 0.0;
            result.contact = start;
            return result;
        }
        alpha = probe_alpha;
    }

    StaticContact current = swept_configuration(
        p10, p20, dp1, dp2, center, radius, tolerance, alpha);
    double previous_alpha = alpha;
    StaticContact previous = current;

    // PHYS-APPROX: endpoint-linear swept CCD is used only to estimate a retry dt;
    // ceiling = strongly curved within-step trajectories can shift the estimated event time;
    // upgrade = evaluate the same contact query over RKF stage trajectories.
    for (int iteration = 0; iteration < 256 && alpha < 1.0; ++iteration) {
        double step = 0.1;
        if (current.gap > tolerance)
            step = 0.8 * (current.gap - tolerance) / speed_bound;
        step = fmax(1.0e-8, fmin(0.1, step));
        previous_alpha = alpha;
        previous = current;
        alpha = fmin(1.0, alpha + step);
        current = swept_configuration(
            p10, p20, dp1, dp2, center, radius, tolerance, alpha);

        if (current.gap <= tolerance) {
            double lo = previous_alpha;
            double hi = alpha;
            for (int bisect = 0; bisect < 64; ++bisect) {
                double mid = 0.5 * (lo + hi);
                StaticContact trial = swept_configuration(
                    p10, p20, dp1, dp2, center, radius, tolerance, mid);
                if (trial.gap <= tolerance) hi = mid;
                else lo = mid;
            }
            result.hit = true;
            result.alpha = hi;
            result.contact = swept_configuration(
                p10, p20, dp1, dp2, center, radius, tolerance, hi);
            return result;
        }
    }
    return result;
}

KOKKOS_INLINE_FUNCTION
Vec3 common_tangent_velocity(const Vec3& free_velocity,
                             const Vec3& glide_normal_raw,
                             const Vec3& sphere_normal_raw,
                             double tangent_tolerance,
                             bool* mobile)
{
    Vec3 glide_normal = glide_normal_raw.normalized();
    Vec3 sphere_normal = sphere_normal_raw.normalized();
    Vec3 tangent = cross(glide_normal, sphere_normal);
    double tangent2 = tangent.norm2();
    if (tangent2 <= tangent_tolerance * tangent_tolerance) {
        if (mobile) *mobile = false;
        return Vec3(0.0);
    }
    tangent = (1.0 / sqrt(tangent2)) * tangent;
    if (mobile) *mobile = true;
    return dot(free_velocity, tangent) * tangent;
}

} // namespace OrowanGeometry
} // namespace ExaDiS

#endif
