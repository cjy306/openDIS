/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "types.h"
#include "crystal.h"
#include "params.h"
#include "oprec.h"

#pragma once
#ifndef EXADIS_SYSTEM_H
#define EXADIS_SYSTEM_H

#include <vector>

namespace ExaDiS {

extern FILE* flog;

/*---------------------------------------------------------------------------
 *
 *    Struct:        SphericalObstacle
 *                  Spherical precipitate for Orowan bypass.
 *                  All units in Burgers vector magnitude.
 *
 *-------------------------------------------------------------------------*/
struct SphericalObstacle {
    Vec3   center;
    double radius;
    int    id;
};

/*---------------------------------------------------------------------------
 *
 *    Struct:        PlanarObstacle
 *                  Planar obstacle (e.g. twin boundary) that blocks
 *                  dislocation motion.  Defined by one point on the plane
 *                  and the outward unit normal.
 *                  All units in Burgers vector magnitude.
 *
 *-------------------------------------------------------------------------*/
struct PlanarObstacle {
    Vec3   point;   // any point on the plane
    Vec3   normal;  // unit normal
    int    id;
};

class System {
public:
    DisNetManager* net_mngr = nullptr;
    double neighbor_cutoff;
    
    inline SerialDisNet* get_serial_network() { return net_mngr->get_serial_network(); }
    inline DeviceDisNet* get_device_network() { return net_mngr->get_device_network(); }
    
    inline int Nnodes_local() { return net_mngr->Nnodes_local(); }
    inline int Nsegs_local() { return net_mngr->Nsegs_local(); }
    
    inline int Nnodes_total() { return Nnodes_local(); }
    inline int Nsegs_total() { return Nsegs_local(); }
    
    T_x xold;
    
    Mat33 extstress;
    double realdt;
    Mat33 dEp, dWp;
    double density;
    
    Params params;
    Crystal crystal;
    
    System();
    ~System();
    System(const System&) = delete;
    void initialize(Params _params, Crystal _crystal, SerialDisNet* network);
    void register_neighbor_cutoff(double cutoff);
    void plastic_strain();
    void reset_glide_planes();
    void write_config(std::string filename);
    
    OpRec* oprec = nullptr;
    
    int num_ranks;
    int proc_rank;
    
    struct SystemTimer {
        Kokkos::Timer timer;
        double accumtime;
        std::string label;
        SystemTimer() { accumtime = 0.0; }
        SystemTimer(std::string _label) : label(_label) { accumtime = 0.0; }
        void start() { timer.reset(); }
        void stop() { accumtime += timer.seconds(); }
    };
    enum timers {TIMER_FORCE, TIMER_MOBILITY, TIMER_INTEGRATION, TIMER_CROSSSLIP,
                 TIMER_COLLISION, TIMER_TOPOLOGY, TIMER_REMESH, TIMER_OUTPUT, TIMER_END};
    SystemTimer timer[TIMER_END];
    
    // Spherical precipitates for Orowan bypass (all units in Burgers vector)
    std::vector<SphericalObstacle> obstacles;

    void load_obstacles(const std::vector<Vec3>& centers_b,
                        const std::vector<double>& radii_b)
    {
        obstacles.clear();
        int n = (int)centers_b.size();
        obstacles.reserve(n);
        for (int i = 0; i < n; i++)
            obstacles.push_back({centers_b[i], radii_b[i], i});
        ExaDiS_log("[System] %d spherical obstacles loaded\n", n);
    }

    // Planar obstacles (twin boundaries) (all units in Burgers vector)
    std::vector<PlanarObstacle> planar_obstacles;

    void load_planar_obstacles(const std::vector<Vec3>& points_b,
                               const std::vector<Vec3>& normals_b)
    {
        planar_obstacles.clear();
        int n = (int)points_b.size();
        planar_obstacles.reserve(n);
        for (int i = 0; i < n; i++) {
            Vec3 nn = normals_b[i];
            double len = sqrt(nn.norm2());
            if (len > 1e-15) nn = (1.0/len) * nn;
            planar_obstacles.push_back({points_b[i], nn, i});
        }
        ExaDiS_log("[System] %d planar obstacles (twin boundaries) loaded\n", n);
    }

    bool pyexadis = false;
    static const int MAX_DEV_TIMERS = 20;
    int numdevtimer = 0;
    SystemTimer devtimer[MAX_DEV_TIMERS];
    int add_timer(std::string label) {
        if (pyexadis) return 0;
        if (numdevtimer == MAX_DEV_TIMERS)
            ExaDiS_fatal("Error: MAX_DEV_TIMERS = %d limit reached\n", MAX_DEV_TIMERS);
        devtimer[numdevtimer++].label = label;
        return numdevtimer-1;
    }
    void print_timers(double timetot=-1.0, bool dev=false);
};

System* make_system(SerialDisNet* net, Crystal crystal=Crystal(), Params params=Params());
DisNetManager* make_network_manager(SerialDisNet* net);

} // namespace ExaDiS

#endif
