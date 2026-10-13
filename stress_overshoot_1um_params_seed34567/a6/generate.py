"""Generate the same physical seed34567 layout with case-specific discretization."""
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
import overshoot_common
overshoot_common.MAXSEG_B = 200
if __name__ == '__main__':
    overshoot_common.run_cli('A75', 34567, BASE_DIR)
