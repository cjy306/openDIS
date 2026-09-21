"""Convert both coherency-validation trajectories to VTK."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for relative_path in ("core/exadis/python", "core/pydis/python", "python", "lib"):
    path = str(REPO_ROOT / relative_path)
    if path not in sys.path:
        sys.path.append(path)


def _step_number(path: Path) -> int:
    match = re.fullmatch(r"config\.(\d+)\.data", path.name)
    if match is None:
        raise ValueError(f"not a numeric configuration snapshot: {path}")
    return int(match.group(1))


def _convert_case(input_dir: Path, output_dir: Path) -> None:
    from pyexadis_utils import read_paradis, write_vtk

    if not input_dir.is_dir():
        raise FileNotFoundError(f"missing simulation output directory: {input_dir}")
    snapshots = sorted(input_dir.glob("config.*.data"), key=_step_number)
    if not snapshots:
        raise FileNotFoundError(f"no configuration snapshots in: {input_dir}")
    output_dir.mkdir(exist_ok=True)
    for index, data_file in enumerate(snapshots, start=1):
        vtk_file = output_dir / data_file.with_suffix(".vtk").name
        write_vtk(
            read_paradis(str(data_file)), str(vtk_file),
            crystal="BCC", verbose=False,
        )
        print(f"[{index}/{len(snapshots)}] {vtk_file}")


def main() -> None:
    import pyexadis

    pyexadis.initialize()
    try:
        _convert_case(
            SCRIPT_DIR / "output_no_coherency",
            SCRIPT_DIR / "vtk_no_coherency",
        )
        _convert_case(
            SCRIPT_DIR / "output_with_coherency",
            SCRIPT_DIR / "vtk_with_coherency",
        )
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
