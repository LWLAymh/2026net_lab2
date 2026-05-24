from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "trajectory_estimates.csv"
SCENES = ("indoor", "outdoor")


def load_scene_rows(path: Path) -> dict[str, list[dict[str, float]]]:
    scene_rows: dict[str, list[dict[str, float]]] = {scene: [] for scene in SCENES}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row["scene"].strip()
            if scene not in scene_rows:
                continue
            scene_rows[scene].append(
                {
                    "point": int(row["point"]),
                    "raw_error": float(row["raw_error"]),
                    "smooth_error": float(row["smooth_error"]),
                }
            )
    for rows in scene_rows.values():
        rows.sort(key=lambda item: item["point"])
    return scene_rows


def plot_error(scene: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    points = [row["point"] for row in rows]
    raw_errors = [row["raw_error"] for row in rows]
    smooth_errors = [row["smooth_error"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(points, raw_errors, "o-", color="#d62728")
    ax.set_xlabel("point index")
    ax.set_ylabel("estimated - true / m")
    ax.set_title(f"{scene.capitalize()} Experiment: Distance Error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / f"{scene}_raw_error.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(points, smooth_errors, "o-", color="#1f77b4")
    ax.set_xlabel("point index")
    ax.set_ylabel("estimated - true / m")
    ax.set_title(f"{scene.capitalize()} Experiment: Smoothed Error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / f"{scene}_smooth_error.png")
    plt.close(fig)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    scene_rows = load_scene_rows(INPUT)
    for scene in SCENES:
        plot_error(scene, scene_rows[scene])

    print(f"saved: {ROOT / 'indoor_raw_error.png'}")
    print(f"saved: {ROOT / 'indoor_smooth_error.png'}")
    print(f"saved: {ROOT / 'outdoor_raw_error.png'}")
    print(f"saved: {ROOT / 'outdoor_smooth_error.png'}")


if __name__ == "__main__":
    main()
