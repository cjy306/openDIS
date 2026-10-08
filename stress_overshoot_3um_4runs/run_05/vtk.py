"""3 um FR, seed 56789."""
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from vtk_common import convert
SEED = 56789
if __name__ == '__main__':
    convert(SEED, BASE_DIR)
