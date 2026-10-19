"""Resolve the existing OpenDiS installation without modifying it."""
import os
import sys
from pathlib import Path

def setup():
    candidates = ([Path(os.environ['OPENDIS_ROOT'])] if 'OPENDIS_ROOT' in os.environ else [])
    candidates += list(Path(__file__).resolve().parents)
    candidates += [Path.home() / 'openDIS', Path('E:/openDIS')]
    for root in candidates:
        if (root / 'core/exadis/python/pyexadis_base.py').is_file():
            for part in ('python', 'lib', 'core/pydis/python', 'core/exadis/python'):
                sys.path.insert(0, str(root / part))
            return root
    raise FileNotFoundError('Set OPENDIS_ROOT to your OpenDiS repository')


#!/usr/bin/env python3
"""Convert saved 1 um FR configurations (seed 12345) to FCC VTK files."""
import argparse
from pathlib import Path
import re
import sys

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
SEED = 12345


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--stride', type=int, default=1,
                        help='Convert every Nth saved snapshot after filtering')
    parser.add_argument('--input', default='output')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    if args.end is not None and args.end < args.start:
        parser.error('--end must be at least --start')
    input_dir = BASE_DIR / args.input
    snapshots = []
    for path in input_dir.glob('config.*.data'):
        match = re.fullmatch(r'config\.(\d+)\.data', path.name)
        if match:
            step = int(match.group(1))
            if step >= args.start and (args.end is None or step <= args.end):
                snapshots.append((step, path))
    snapshots = sorted(snapshots)[::args.stride]
    if not snapshots:
        raise FileNotFoundError(f'No matching config.*.data in {input_dir}')

    setup()
    import pyexadis
    from pyexadis_utils import read_paradis, write_vtk

    output_dir = BASE_DIR / ('vtk_' + Path(args.input).name)
    output_dir.mkdir(exist_ok=True)
    pyexadis.initialize()
    try:
        for index, (step, data_file) in enumerate(snapshots, 1):
            network = read_paradis(str(data_file))
            vtk_file = output_dir / f'config.{step:010d}.vtk'
            write_vtk(network, str(vtk_file), crystal='FCC', verbose=False)
            print(f'[{index}/{len(snapshots)}] {vtk_file}', flush=True)
    finally:
        pyexadis.finalize()


if __name__ == '__main__':
    main()
