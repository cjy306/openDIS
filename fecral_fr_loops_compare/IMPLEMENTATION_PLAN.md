# FeCrAl FR/loop comparison implementation plan

> Execute inline using executing-plans and test-driven-development. User approved implementation.

**Goal:** Paired FR-only / FR+loop cases with relaxation, loading, plots, VTK and two submissions per case.
**Architecture:** Shared generator/relaxation helpers; explicit per-case test settings; OpenDiS remains read-only.
**Tech Stack:** Python, NumPy, Matplotlib, ExaDiS, Slurm.
**Spec:** Approved chat parameters, recorded in README.md.

## Constraints and review focus
All files under D:/Wworkspace/fecral_fr_loops_compare. No core edits or job submissions.
1um PBC, rho1e14, FR0.4um, seed12345; 1000 111 hexagonal loops only in with_loops.
Check paired FR identity, actual line density, loop geometry and slip planes.
Reuse 10ns/1e-7 net tensor relaxation; failed convergence blocks loading.
BCC_NL300K is pure-Fe approximation; keep glide constraints and cross-slip off.
Plot validates units and monotonic strain; native runtime must be tested on cluster if unavailable locally.

## Tasks
- [ ] Tests first: layout, relaxation windows, plot parser.
- [ ] Generator, reused relaxation, per-case tests and VTK.
- [ ] Comparison plot, four submissions, README.
- [ ] Offline tests, syntax checks, geometry checks and limitations report.

## Rulings
Independent user-requested artifact directory, no git operations in source repo.
User selected Kunshan: ksagnormal01, one GPU, eight CPU cores, existing conda/module template.

## Verification 2026-10-27
Four offline tests passed including actual Python geometry helpers; all nine Python files parsed; four bash -n checks passed.
Native pyexadis is unavailable locally. No full relaxation/loading or physical-convergence claim.
Files created outside source repository as requested. No source core modified and no jobs submitted.
