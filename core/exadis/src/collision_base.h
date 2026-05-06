/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Minimal base class for collision modules.
 *	Kept in a separate header to avoid circular includes between
 *	collision.h (aggregator) and the individual collision-type headers.
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_COLLISION_BASE_H
#define EXADIS_COLLISION_BASE_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Collision
 *
 *-------------------------------------------------------------------------*/
class Collision {
public:
    Collision() {}
    Collision(System *system) {}
    virtual void handle(System *system) {}
    virtual void pre_integrate(System *system) {}
    virtual void post_remesh(System *system) {}
    virtual const char* name() { return "CollisionNone"; }
};

} // namespace ExaDiS

#endif
