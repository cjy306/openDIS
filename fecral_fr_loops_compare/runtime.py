"""Locate the existing OpenDiS installation; never modify it."""
import os
import sys
from pathlib import Path

def setup():
    roots = ([Path(os.environ['OPENDIS_ROOT'])] if 'OPENDIS_ROOT' in os.environ else [])
    roots += list(Path(__file__).resolve().parents)
    roots += [Path.home() / 'openDIS', Path('E:/openDIS')]
    for root in roots:
        if (root / 'core/exadis/python/pyexadis_base.py').is_file():
            for part in ('python', 'lib', 'core/pydis/python', 'core/exadis/python'):
                path = str(root / part)
                if path not in sys.path:
                    sys.path.insert(0, path)
            return root
    raise FileNotFoundError('Set OPENDIS_ROOT to your OpenDiS installation')
