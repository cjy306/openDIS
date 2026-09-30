"""Convert saved ExaDiS configurations using the repository's FCC VTK writer."""
import argparse
from pathlib import Path
import re
import sys


def selected_snapshots(directory, start=0, end=None, stride=1):
    snapshots = []
    for path in directory.glob('config.*.data'):
        match = re.fullmatch(r'config\.(\d+)\.data', path.name)
        if match:
            step = int(match.group(1))
            if step >= start and (end is None or step <= end):
                snapshots.append((step, path))
    return sorted(snapshots)[::stride]


def run_cli(seed, base_dir):
    parser = argparse.ArgumentParser(description='Convert one FCC run to VTK')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--stride', type=int, default=1,
                        help='Convert every Nth saved snapshot after filtering')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    directory = Path(base_dir)
    snapshots = selected_snapshots(directory / f'output_A75_seed{seed}',
                                   args.start, args.end, args.stride)
    if not snapshots:
        raise FileNotFoundError(f'No matching config.*.data under {directory}')
    repo = Path(__file__).resolve().parent.parent
    for path in (repo / 'python', repo / 'lib', repo / 'core/pydis/python',
                 repo / 'core/exadis/python'):
        sys.path.append(str(path))
    import pyexadis
    from pyexadis_utils import read_paradis, write_vtk
    output = directory / f'vtk_A75_seed{seed}'
    output.mkdir(exist_ok=True)
    pyexadis.initialize()
    try:
        for index, (step, path) in enumerate(snapshots, 1):
            network = read_paradis(str(path))
            target = output / f'config.{step:010d}.vtk'
            write_vtk(network, str(target), crystal='FCC', verbose=False)
            print(f'[{index}/{len(snapshots)}] {target}', flush=True)
    finally:
        pyexadis.finalize()

