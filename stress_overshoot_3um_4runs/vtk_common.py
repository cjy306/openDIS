"""Convert one completed run's saved configurations to FCC VTK."""
import argparse
from pathlib import Path
import re
from common import setup

def convert(seed, directory):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1 or (args.end is not None and args.end < args.start):
        parser.error('Invalid stride or step range')
    directory = Path(directory)
    snapshots = []
    for path in (directory/f'output_A75_3um_seed{seed}').glob('config.*.data'):
        match = re.fullmatch(r'config\.(\d+)\.data',path.name)
        if match:
            step = int(match.group(1))
            if step >= args.start and (args.end is None or step <= args.end):
                snapshots.append((step,path))
    snapshots = sorted(snapshots)[::args.stride]
    if not snapshots:
        raise FileNotFoundError(f'No matching snapshots under {directory}')
    setup()
    import pyexadis
    from pyexadis_utils import read_paradis, write_vtk
    output = directory/f'vtk_A75_3um_seed{seed}'
    output.mkdir(exist_ok=True)
    pyexadis.initialize()
    try:
        for index,(step,path) in enumerate(snapshots,1):
            net = read_paradis(str(path))
            target = output/f'config.{step:010d}.vtk'
            write_vtk(net,str(target),crystal='FCC',verbose=False)
            print(f'[{index}/{len(snapshots)}] {target}',flush=True)
    finally:
        pyexadis.finalize()
