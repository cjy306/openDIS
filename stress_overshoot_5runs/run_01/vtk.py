"""Convert one FCC Cu run to VTK."""
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from vtk_common import run_cli

SEED = 12345

if __name__ == "__main__":
    run_cli(SEED, BASE_DIR)
