# 1 um FR parameter comparison (seed 34567)

| Case | a | rtol | rann | maxseg | minseg |
|---|---:|---:|---:|---:|---:|
| Existing baseline | 4 | 1 | 2 | 200 | 40 |
| a6 | 6 | 1.5 | 3 | 200 | 40 |
| seg400 | 4 | 1 | 2 | 400 | 100 |

All length parameters are in Burgers-vector units (b=0.255 nm).
Both cases preserve the original 128 sources, 1 um length, 5 um periodic box,
A75 slip allocation, random centers and angles at seed34567.
The seg400 generator changes only discretization: 11 nodes/source rather than 21.
Elasticity, mobility, strain rate 1000/s, final strain 0.5%, cross slip off,
write_freq=1000 and the other simulation settings are copied from the baseline.
These are two coupled parameter tests, not individual-parameter attribution.
Default core energy changes with a; automatic subcycling groups can also change.

Run in this folder on Kunshan:

    sbatch submit_test_kun_a6.sh
    sbatch submit_test_kun_seg400.sh

After snapshots are available, submit conversion separately:

    sbatch submit_vtk_kun_a6.sh
    sbatch submit_vtk_kun_seg400.sh

All submissions request 1 GPU, 8 CPUs, 120 hours and exclude gnode36.
Simulation submissions execute generate then test and start fresh.
For a restart, use test.py --restart STEP separately without regeneration.

    python plot.py

Plot reads the existing baseline from ../stress_overshoot_5runs/run_03,
and new results from a6 and seg400. It saves PNG under plots/.
Missing cases are reported, partial curves are labeled and never extrapolated.
Original simulation folders are not modified.
