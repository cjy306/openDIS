"""3 um FR, seed 23456."""
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from common import simulate
SEED = 23456
if __name__ == '__main__':
    simulate(SEED, BASE_DIR)
