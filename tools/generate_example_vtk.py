#!/usr/bin/env python3
import os, sys
from importlib import util

# load pyexadis_utils module by path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core', 'exadis', 'python', 'pyexadis_utils.py'))
spec = util.spec_from_file_location('pyexadis_utils', module_path)
py = util.module_from_spec(spec)
spec.loader.exec_module(py)

# generate a small BCC line configuration and write vtk
G = py.generate_line_config('BCC', 10.0, num_lines=24, seed=42)
# wrap into DisNetManager
N = py.DisNetManager(G)

outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(outdir, exist_ok=True)
vtkfile = os.path.join(outdir, 'example_slips.vtk')

py.write_vtk(N, vtkfile, crystal='BCC', add_slipsystems=True)
print('Wrote example VTK to', vtkfile)
