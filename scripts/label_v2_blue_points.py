from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "positioning_results_v2"
OUT = ROOT / "positioning_results_v2_labeled"

INDOOR_POINTS = np.array(
    [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.5],
        [2.0, 3.0],
        [3.5, 3.0],
        [3.5, 4.2],
        [1.0, 3.2],
        [1.0, 1.6],
        [3.0, 1.3],
        [4.0, 1.0],
    ]
)

OUTDOOR_POINTS = np.array(
    [
        [9.6, 4.2],
        [7.2, 5.4],
        [4.8, 6.6],
        [1.2, 8.4],
        [1.2, 4.8],
        [1.2, 1.2],
        [6.0, 1.2],
        [7.2, 1.2],
        [9.6, 1.2],
        [9.6, 5.4],
    ]
)

INDOOR_ANCHORS = {
    "ymh": np.array([0.0, 4.2]),
    "wkx": np.array([4.0, 0.0]),
    "whx": np.array([4.0, 4.2]),
}

OUTDOOR_ANCHORS = {
    "ymh": np.array([0.0, 7.2]),
    "wkx": np.array([9.6, 0.0]),
    "whx": np.array([0.0, 0.0]),
}

PILLAR = (6.2, 0.5, 6.85 - 6.2, 1.1 - 0.5)


def weighted_smooth(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return points.copy()
    smoothed = points.copy()
    for i in range(1, len(points) - 1):
        smoothed[i] = 0.2 * points[i - 1] + 0.6 * points[i] + 0.2 * points[i + 1]
    return smoothed


def read_trajectory(scene: str) -> np.ndarray:
    rows = []
    with (SRC / "trajectory_estimates.csv").open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["scene"] == scene:
                rows.append(row)
    rows.sort(key=lambda row: int(row["point"]))
    return np.array([[float(row["est_x"]), float(row["est_y"])] for row in rows])


def plot_scene(
    scene: str,
    true_points: np.ndarray,
    raw: np.ndarray,
    smoothed: np.ndarray,
    anchors: dict[str, np.ndarray],
    pillar: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    ax.plot(true_points[:, 0], true_points[:, 1], "o-", color="black", linewidth=2, markersize=4, label="true path")
    ax.plot(raw[:, 0], raw[:, 1], "o--", color="#e07a2f", linewidth=1.2, markersize=3, alpha=0.55, label="estimated raw")
    ax.plot(smoothed[:, 0], smoothed[:, 1], "o-", color="#1f77b4", linewidth=2, markersize=4, label="estimated smoothed")

    for idx, (x, y) in enumerate(true_points):
        ax.text(x + 0.06, y + 0.06, f"P{idx}", fontsize=7, color="black")
    for idx, (x, y) in enumerate(raw):
        ax.text(x + 0.05, y - 0.10, f"{idx}", fontsize=6, color="#c65f1a", alpha=0.9)
    for idx, (x, y) in enumerate(smoothed):
        ax.text(x + 0.05, y + 0.06, f"S{idx}", fontsize=6, color="#1f5fa8", alpha=0.95)

    for name, xy in anchors.items():
        ax.scatter([xy[0]], [xy[1]], marker="^", s=90, label=f"{name} anchor")
        ax.text(xy[0] + 0.08, xy[1] + 0.08, name)

    if pillar:
        ax.add_patch(Rectangle((PILLAR[0], PILLAR[1]), PILLAR[2], PILLAR[3], facecolor="#777777", edgecolor="black", alpha=0.35, label="pillar"))

    ax.set_title(f"{scene.capitalize()} 2D Positioning")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"{scene}_trajectory_labeled.png")
    plt.close(fig)


def plot_linear() -> None:
    rows = []
    with (SRC / "linear_estimates.csv").open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(key=lambda row: int(row["point"]))

    true_d = np.array([float(row["true_distance"]) for row in rows])
    estimated = np.array([float(row["estimated_distance"]) for row in rows])
    smoothed = np.array([float(row["smoothed_distance"]) for row in rows])

    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=180)
    ax.plot([0, 24], [0, 24], color="black", linewidth=1.2, label="ideal")
    ax.scatter(true_d, estimated, color="#e07a2f", label="estimated")
    ax.plot(true_d, smoothed, "o-", color="#1f77b4", label="smoothed")
    for idx, (x, y) in enumerate(zip(true_d, smoothed)):
        ax.text(x + 0.12, y + 0.20, f"S{idx}", fontsize=6, color="#1f5fa8", alpha=0.95)
    ax.set_xlabel("true distance / m")
    ax.set_ylabel("estimated distance / m")
    ax.set_title("Linear Experiment: Estimated vs True")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "linear_estimated_vs_true_labeled.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    indoor_raw = read_trajectory("indoor")
    outdoor_raw = read_trajectory("outdoor")
    plot_scene("indoor", INDOOR_POINTS, indoor_raw, weighted_smooth(indoor_raw), INDOOR_ANCHORS)
    plot_scene("outdoor", OUTDOOR_POINTS, outdoor_raw, weighted_smooth(outdoor_raw), OUTDOOR_ANCHORS, pillar=True)
    plot_linear()
    print(f"labeled v2 plots: {OUT}")


if __name__ == "__main__":
    main()
