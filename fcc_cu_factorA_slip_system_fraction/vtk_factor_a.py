#!/usr/bin/env python3
"""将A25、A50、A75、A100四组网络快照批量转换为VTK。"""

from pathlib import Path
import re
import sys


SEED = 12345
START = 0
END = None
CASES = ("A25", "A50", "A75", "A100")

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
for path in (REPO_ROOT / "python", REPO_ROOT / "lib",
             REPO_ROOT / "core" / "pydis" / "python",
             REPO_ROOT / "core" / "exadis" / "python"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import pyexadis
from pyexadis_utils import read_paradis, write_vtk


def step_number(path):
    match = re.fullmatch(r"config\.(\d+)\.data", path.name)
    return int(match.group(1)) if match else None


def selected_snapshots(input_dir):
    snapshots = []
    for path in input_dir.glob("config.*.data"):
        step = step_number(path)
        if step is None:
            continue
        if START is not None and step < START:
            continue
        if END is not None and step > END:
            continue
        snapshots.append((step, path))
    return sorted(snapshots, key=lambda item: item[0])


def main():
    jobs = []
    for case in CASES:
        input_dir = BASE_DIR / f"output_{case}_seed{SEED}"
        snapshots = selected_snapshots(input_dir)
        if not snapshots:
            print(f"[跳过] 没有快照: {input_dir}")
            continue
        output_dir = BASE_DIR / f"vtk_{case}_seed{SEED}"
        jobs.append((case, snapshots, output_dir))

    if not jobs:
        raise FileNotFoundError("四组输出目录中均未找到config.*.data")

    failures = []
    pyexadis.initialize()
    try:
        for case, snapshots, output_dir in jobs:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{case}] 转换 {len(snapshots)} 个快照")
            for index, (step, data_file) in enumerate(snapshots, start=1):
                vtk_file = output_dir / f"config.{step}.vtk"
                try:
                    network = read_paradis(str(data_file))
                    write_vtk(network, str(vtk_file), crystal="FCC", verbose=False)
                    print(f"  [{index}/{len(snapshots)}] {vtk_file.name}")
                except Exception as error:
                    failures.append((data_file, error))
                    print(f"  [失败] {data_file}: {error}")
    finally:
        pyexadis.finalize()

    if failures:
        raise RuntimeError(f"共有 {len(failures)} 个快照转换失败")
    print("四组VTK转换完成")


if __name__ == "__main__":
    main()
