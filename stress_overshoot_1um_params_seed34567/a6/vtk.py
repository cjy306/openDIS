"""Convert this case's saved configurations to FCC VTK."""
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from vtk_common import run_cli
if __name__ == '__main__':
    run_cli(34567, BASE_DIR)
