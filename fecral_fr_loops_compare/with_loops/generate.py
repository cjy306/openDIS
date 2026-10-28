"""Generate the paired FR background plus irradiation loops for this case."""
from pathlib import Path
import sys
import faulthandler

if __name__ == '__main__':
    faulthandler.enable(all_threads=True)
    print(f'[generate] entry: {__file__}; python={sys.executable}', flush=True)
    print('[generate] importing shared generator (including NumPy)', flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import main

if __name__ == '__main__':
    print('[generate] shared generator imported', flush=True)
    main(default_case='with_loops')
