"""3 um FR, seed 34567."""
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from common import generate
SEED = 34567
if __name__ == '__main__':
    generate(SEED, BASE_DIR)
