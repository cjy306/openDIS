"""Offline geometry/parameter checks; does not run the ExaDiS solver."""
import ast
from collections import Counter
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import overshoot_common as cfg

ROOT = Path(__file__).resolve().parent


def main():
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    utility = ROOT.parent / 'core/exadis/python/pyexadis_utils.py'
    tree = ast.parse(utility.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'insert_frank_read_src')
    scope = {'np': np, 'NodeConstraints': SimpleNamespace(
        PINNED_NODE=7, UNCONSTRAINED=0)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(utility), 'exec'), scope)
    previous = None
    normalized = {}
    for run, seed in enumerate((12345, 23456, 34567, 45678, 56789), 1):
        directory = ROOT / f'run_{run:02d}'
        for filename in ('generate.py', 'test.py'):
            source = (directory / filename).read_text(encoding='utf-8')
            assert f'SEED = {seed}' in source
            source = source.replace(f'SEED = {seed}', 'SEED = SEED_VALUE')
            if filename in normalized:
                assert source == normalized[filename]
            normalized[filename] = source
        test = (directory / 'test.py').read_text(encoding='utf-8')
        assert 'max_strain=0.005' in test and 'cross_slip=None' in test
        assert 'erate=1e3' in test
        positions, systems = cfg.make_layout(seed, 'A75')
        assert (positions, systems) == cfg.make_layout(seed, 'A75')
        assert positions != previous
        previous = positions
        counts = Counter(systems)
        assert all(counts[s] == 12 for s in cfg.HIGH_SCHMID_001)
        assert all(counts[s] == 8 for s in cfg.ZERO_SCHMID_001)
        rng = random.Random(seed ^ 0x6C8E9CF5)
        nodes, segments = [], []
        for center, system in zip(positions, systems):
            b = np.array(cfg._FCC_B[system]) / np.sqrt(2)
            n = np.array(cfg._FCC_N[system]) / np.sqrt(3)
            schmid = abs(b[2] * n[2])
            assert np.isclose(schmid, 1 / np.sqrt(6) if system in cfg.HIGH_SCHMID_001 else 0)
            theta = rng.uniform(0, 180)
            nodes, segments = scope['insert_frank_read_src'](
                None, nodes, segments, b, n, cfg.FR_LENGTH_M / cfg.BURGMAG_M,
                np.array(center) * cfg.LBOX_M / cfg.BURGMAG_M,
                theta=theta, numnodes=21)
        xyz = np.array(nodes)[:, :3]
        seg = np.array(segments)
        vectors = xyz[seg[:, 1].astype(int)] - xyz[seg[:, 0].astype(int)]
        assert np.allclose(np.sum(vectors * seg[:, 5:8], axis=1), 0, atol=1e-10)
        rho = np.linalg.norm(vectors, axis=1).sum() * cfg.BURGMAG_M / cfg.LBOX_M**3
        assert np.isclose(rho, 1.024e12, rtol=1e-12)
        assert np.linalg.norm(vectors, axis=1).max() <= cfg.MAXSEG_B
        assert sum(row[3] == 7 for row in nodes) == 256
        print(f'run_{run:02d}: seed={seed}, sources=128, high/zero=96/32, rho={rho:.12g}: PASS')
    print('Syntax, seed-only differences, reproducibility, FCC geometry and density: PASS')


if __name__ == '__main__':
    main()
