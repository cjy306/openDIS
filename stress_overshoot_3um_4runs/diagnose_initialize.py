"""Reproduce generator imports and initialization without creating a network."""
import sys
import subprocess

from common import setup

setup()
print('Python:', sys.executable, sys.version, flush=True)
print('Importing generator dependencies', flush=True)
import numpy as np
import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import insert_frank_read_src, write_vtk

print('NumPy:', np.__version__, np.__file__, flush=True)
print('pyexadis:', pyexadis.__file__, flush=True)
subprocess.run(['ldd', pyexadis.__file__], check=False)
print('BEFORE pyexadis.initialize()', flush=True)
pyexadis.initialize()
print('AFTER pyexadis.initialize()', flush=True)
pyexadis.finalize()
print('AFTER pyexadis.finalize(): initialization check passed', flush=True)
