"""Convert relaxation and loading snapshots, with BCC Burgers-vector labels."""
import argparse
from pathlib import Path
import re
import sys
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from runtime import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    folder = BASE / args.input
    numbered = []
    for path in folder.glob("config.*.data"):
        match = re.fullmatch(r"config\.(\d+)\.data",path.name)
        if match:
            numbered.append((int(match[1]),path))
    paths = [p for _,p in sorted(numbered)][::args.stride]
    paths += [folder/name for name in ("relaxed_config.data","loading_start.data","loading_final.data")
              if (folder/name).exists()]
    if not paths:
        raise FileNotFoundError(f"No snapshots in {folder}")
    setup()
    import pyexadis
    from pyexadis_utils import read_paradis, write_vtk
    target = BASE/("vtk_"+folder.name)
    target.mkdir(exist_ok=True)
    pyexadis.initialize()
    try:
        for path in paths:
            write_vtk(read_paradis(str(path)),str(target/(path.stem+".vtk")),
                      crystal="BCC",verbose=False)
            print(target/(path.stem+".vtk"),flush=True)
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
