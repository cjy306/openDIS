#!/usr/bin/env python3
import numpy as np
import os

outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(outdir, exist_ok=True)
vtkfile = os.path.join(outdir, 'example_slips_demo.vtk')

# cube cell origin and lattice vectors
cell_origin = np.array([0.0,0.0,0.0])
h0 = np.array([10.0,0.0,0.0])
h1 = np.array([0.0,10.0,0.0])
h2 = np.array([0.0,0.0,10.0])

c = cell_origin + np.array([np.zeros(3), h0, h1, h2, h0+h1, h0+h2, h1+h2, h0+h1+h2])

# define 6 segments with endpoints
rsegs = np.array([
    [2.0,2.0,2.0, 6.0,2.0,2.0],   # segment 0
    [2.0,4.0,2.0, 6.0,4.0,2.0],   # segment 1
    [2.0,6.0,2.0, 6.0,6.0,2.0],   # segment 2
    [2.0,2.0,6.0, 6.0,2.0,6.0],   # segment 3
    [2.0,4.0,6.0, 6.0,4.0,6.0],   # segment 4
    [2.0,6.0,6.0, 6.0,6.0,6.0],   # segment 5
])

nsegs = rsegs.shape[0]

# Burgers and plane vectors (dummy values)
b = np.array([
    [1.0,1.0,1.0],
    [-1.0,1.0,1.0],
    [1.0,-1.0,1.0],
    [1.0,1.0,-1.0],
    [0.0,1.0,1.0],
    [1.0,0.0,1.0]
])

p = np.array([
    [0.0,-1.0,1.0],
    [0.0,1.0,1.0],
    [1.0,0.0,1.0],
    [1.0,1.0,0.0],
    [1.0,1.0,2.0],
    [2.0,1.0,1.0]
])

# slip system ids to demonstrate mapping: include examples from 1,12,13,24,25
SlipSystem = np.array([1, 12, 13, 24, 25, 5], dtype=float)
# CharType: 1=screw,2=mixed,3=edge
CharType = np.array([1,2,3,1,2,3], dtype=float)
CharAngle = np.array([5.0,45.0,85.0,2.0,60.0,88.0], dtype=float)

with open(vtkfile, 'w') as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Example dislocation config\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")

    f.write("POINTS %d FLOAT\n" % (c.shape[0] + 2*nsegs))
    for row in c:
        f.write("%f %f %f\n" % (row[0], row[1], row[2]))
    for i in range(nsegs):
        f.write("%f %f %f\n" % (rsegs[i,0], rsegs[i,1], rsegs[i,2]))
        f.write("%f %f %f\n" % (rsegs[i,3], rsegs[i,4], rsegs[i,5]))

    f.write("CELLS %d %d\n" % (1 + nsegs, 9 + 3*nsegs))
    f.write("8 0 1 4 2 3 5 7 6\n")
    for i in range(nsegs):
        f.write("2 %d %d\n" % (8 + 2*i, 8 + 2*i + 1))

    f.write("CELL_TYPES %d\n" % (1 + nsegs))
    f.write("12\n")
    for i in range(nsegs):
        f.write("4\n")

    f.write("CELL_DATA %d\n" % (1 + nsegs))

    # write Burgers and Planes vectors (first entry zeros for the cube cell)
    f.write("VECTORS Burgers FLOAT\n")
    f.write("%f %f %f\n" % (0.0,0.0,0.0))
    for i in range(nsegs):
        bv = b[i]
        f.write("%f %f %f\n" % (bv[0], bv[1], bv[2]))

    f.write("VECTORS Planes FLOAT\n")
    f.write("%f %f %f\n" % (0.0,0.0,0.0))
    for i in range(nsegs):
        pv = p[i]
        f.write("%f %f %f\n" % (pv[0], pv[1], pv[2]))

    # Scalars
    f.write("SCALARS SlipSystem FLOAT 1\n")
    f.write("LOOKUP_TABLE default\n")
    for v in SlipSystem:
        f.write("%f\n" % v)

    f.write("SCALARS CharType FLOAT 1\n")
    f.write("LOOKUP_TABLE default\n")
    for v in CharType:
        f.write("%f\n" % v)

    f.write("SCALARS CharAngle FLOAT 1\n")
    f.write("LOOKUP_TABLE default\n")
    for v in CharAngle:
        f.write("%f\n" % v)

print('Wrote demo VTK to', vtkfile)
