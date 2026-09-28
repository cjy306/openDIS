"""Generate one FCC Cu A75 random-character initial network."""
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from overshoot_common import run_cli

SEED = 23456

if __name__ == "__main__":
    run_cli("A75", SEED, BASE_DIR)
