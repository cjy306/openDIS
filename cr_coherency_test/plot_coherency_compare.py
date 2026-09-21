"""Compare zero-stress loop evolution with and without coherency stress."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

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


def require_output_files(directory: Path):
    directory = Path(directory)
    property_file = directory / "stress_strain_dens.dat"
    if not property_file.is_file():
        raise FileNotFoundError(f"missing simulation properties: {property_file}")
    snapshots = sorted(directory.glob("config.*.data"), key=_step_number)
    if not snapshots:
        raise FileNotFoundError(f"no configuration snapshots in: {directory}")
    return property_file, snapshots


def unsigned_normal_angle(first, second) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = np.clip(abs(np.dot(first, second)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _load_properties(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().lstrip("#").strip().lower().split()
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] != len(header):
        raise ValueError(f"header/data column mismatch in {path}")
    columns = {name: values[:, index] for index, name in enumerate(header)}
    for required in ("step", "time", "density", "nnodes"):
        if required not in columns:
            raise ValueError(f"missing '{required}' column in {path}; got {header}")
    return columns


def _largest_component(node_count: int, segments: np.ndarray):
    adjacency = [[] for _ in range(node_count)]
    for first, second in segments:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    unseen = set(range(node_count))
    components = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        component = [root]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    return max(components, key=len), adjacency


def _fit_loop_normal(network) -> np.ndarray:
    data = network.export_data()
    positions = np.asarray(data["nodes"]["positions"], dtype=float)
    segments = np.asarray(data["segs"]["nodeids"], dtype=int)
    if len(positions) < 3 or len(segments) < 3:
        raise ValueError("at least three nodes and segments are required to fit a loop plane")
    component, adjacency = _largest_component(len(positions), segments)
    component_set = set(component)
    h = np.asarray(data["cell"]["h"], dtype=float)
    periodic = np.asarray(data["cell"]["is_periodic"], dtype=bool)

    root = component[0]
    unwrapped = {root: positions[root].copy()}
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in component_set or neighbor in unwrapped:
                continue
            fractional_delta = np.linalg.solve(h, positions[neighbor] - positions[node])
            fractional_delta[periodic] -= np.rint(fractional_delta[periodic])
            unwrapped[neighbor] = unwrapped[node] + h @ fractional_delta
            stack.append(neighbor)

    points = np.asarray([unwrapped[index] for index in component])
    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    if eigenvalues[1] <= np.finfo(float).eps * max(1.0, eigenvalues[2]):
        raise ValueError("largest network component is too nearly collinear for a plane fit")
    return eigenvectors[:, 0]


def _rotation_history(snapshot_paths, reference_normal, step_to_time):
    from pyexadis_utils import read_paradis

    times = []
    angles = []
    for path in snapshot_paths:
        step = _step_number(path)
        if step == 0 and step not in step_to_time:
            step_to_time[0] = 0.0
        if step not in step_to_time:
            raise ValueError(f"snapshot step {step} is absent from the property file")
        normal = _fit_loop_normal(read_paradis(str(path)))
        times.append(step_to_time[step])
        angles.append(unsigned_normal_angle(reference_normal, normal))
    return np.asarray(times), np.asarray(angles)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyexadis
    from pyexadis_utils import read_paradis

    cases = (
        ("No coherency", SCRIPT_DIR / "output_no_coherency", "#202020"),
        ("With coherency", SCRIPT_DIR / "output_with_coherency", "#d62728"),
    )
    checked = [(label, *require_output_files(path), color)
               for label, path, color in cases]
    initial_file = SCRIPT_DIR / "init_coherency" / "init_config.data"
    if not initial_file.is_file():
        raise FileNotFoundError(f"missing shared initial configuration: {initial_file}")

    output_dir = SCRIPT_DIR / "post_coherency"
    output_dir.mkdir(exist_ok=True)
    pyexadis.initialize()
    try:
        reference_normal = _fit_loop_normal(read_paradis(str(initial_file)))
        figure, axes = plt.subplots(1, 3, figsize=(16, 5))
        summaries = []
        for label, property_file, snapshots, color in checked:
            columns = _load_properties(property_file)
            steps = columns["step"].astype(int)
            step_to_time = dict(zip(steps, columns["time"]))
            rotation_time, rotation = _rotation_history(
                snapshots, reference_normal, step_to_time
            )
            axes[0].plot(columns["time"], columns["density"], label=label, color=color)
            axes[1].plot(columns["time"], columns["nnodes"], label=label, color=color)
            axes[2].plot(rotation_time, rotation, label=label, color=color, marker="o", ms=2)
            summaries.append((label, rotation[-1], np.nanmax(rotation)))

        axes[0].set_ylabel(r"Dislocation density (m$^{-2}$)")
        axes[1].set_ylabel("Node count")
        axes[2].set_ylabel("Unsigned loop-plane rotation (deg)")
        for axis in axes:
            axis.set_xlabel("Physical time (s)")
            axis.grid(True, linestyle="--", alpha=0.3)
            axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / "coherency_compare.png", dpi=300, bbox_inches="tight")
        plt.close(figure)
        with (output_dir / "coherency_compare_summary.txt").open(
            "w", encoding="utf-8"
        ) as stream:
            for label, final_angle, maximum_angle in summaries:
                stream.write(
                    f"{label}: final_rotation_deg={final_angle:.8e}, "
                    f"max_rotation_deg={maximum_angle:.8e}\n"
                )
    finally:
        pyexadis.finalize()

    print(f"Comparison plot: {output_dir / 'coherency_compare.png'}")


if __name__ == "__main__":
    main()
