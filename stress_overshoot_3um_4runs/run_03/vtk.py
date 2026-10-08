"""3 um FR, seed 34567."""
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from vtk_common import convert
SEED = 34567
if __name__ == '__main__':
    convert(SEED, BASE_DIR)
