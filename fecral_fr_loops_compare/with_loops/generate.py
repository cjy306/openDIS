"""Generate the paired FR background plus irradiation loops for this case."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import main

if __name__ == '__main__':
    main(default_case='with_loops')
