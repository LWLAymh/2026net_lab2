from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
# Keep the current results untouched and write new exploration outputs here.
OUT = Path(os.environ.get("POSITIONING_OUT", str(ROOT / "positioning_results_v4")))


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

LINEAR_DISTANCES = np.array([24.0, 21.6, 19.2, 16.8, 14.4, 12.0, 9.6, 7.2, 4.8, 2.4, 0.0])

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

# Point indices are zero-based and match the P0/P1/... labels in the plots.
# These samples contradict the distance-RSSI trend strongly enough to distort
# the fitted path-loss parameters for that receiver.
MANUAL_FIT_EXCLUDES: dict[tuple[str, str], set[int]] = {
    ("indoor", "ymh"): {1},
    ("indoor", "wkx"): {7, 8, 9},
    ("outdoor", "ymh"): {7},
}

FIT_RESIDUAL_CAPS = {
    "indoor": 4.0,
    "outdoor": 3.0,
    "linear": 4.0,
}
FIT_MIN_SAMPLES = 7
FIT_MAX_ROUNDS = 3

TRAJECTORY_CURVATURE = {
    "indoor": (0.01, 0.02),
    "outdoor": (0.10, 0.20),
}


@dataclass
class FitResult:
    scene: str
    device: str
    a: float
    n: float
    rmse: float
    samples: int
    excluded: str


def path_loss_rssi(distance: np.ndarray | float, a: float, n: float) -> np.ndarray | float:
    distance = np.maximum(np.asarray(distance, dtype=float), 1e-3)
    return a - 10.0 * n * np.log10(distance)


def rssi_to_distance(rssi: np.ndarray | float, a: float, n: float) -> np.ndarray | float:
    return 10.0 ** ((a - np.asarray(rssi, dtype=float)) / (10.0 * n))


def read_summary_values(path: Path, series: str | None = None) -> list[float]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if series is not None and row.get("series") != series:
                continue
            value = row.get("iqr_filtered_mean") or row.get("mean")
            rows.append((row["file"], float(value)))

    def sort_key(item: tuple[str, float]) -> tuple[int, str]:
        stem = Path(item[0]).stem
        return (int(stem) if stem.isdigit() else 10**9, item[0])

    rows.sort(key=sort_key)
    return [value for _, value in rows]


def load_device_data() -> dict[str, dict[str, np.ndarray]]:
    wkx = np.array(read_summary_values(ROOT / "Lab2_wkx" / "rssi_summary.csv"))
    whx = np.array(read_summary_values(ROOT / "Lab2_whx" / "rssi_summary.csv"))
    ymh_indoor = np.array(read_summary_values(ROOT / "Lab2_indoor_ymh" / "rssi_summary.csv", "all"))
    ymh_outdoor = np.array(read_summary_values(ROOT / "Lab2_outdoor_ymh" / "rssi_summary.csv", "all"))
    ymh_linear = np.array(read_summary_values(ROOT / "Lab2_linear_ymh" / "rssi_summary.csv", "all"))

    return {
        "indoor": {
            "wkx": wkx[:10],
            "whx": whx[:10],
            "ymh": ymh_indoor[:10],
        },
        "outdoor": {
            "wkx": wkx[10:20],
            "whx": whx[10:20],
            "ymh": ymh_outdoor[:10],
        },
        "linear": {
            "ymh": ymh_linear[:11],
        },
    }


def fit_path_loss(scene: str, device: str, distances: np.ndarray, rssi: np.ndarray) -> FitResult:
    manual_excludes = MANUAL_FIT_EXCLUDES.get((scene, device), set())
    residual_cap = FIT_RESIDUAL_CAPS.get(scene, 4.0)
    point_indices = np.arange(len(distances))
    base_mask = (distances > 0.15) & np.array([idx not in manual_excludes for idx in point_indices])

    def solve(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = distances[mask]
        y = rssi[mask]

        def residual(params: np.ndarray) -> np.ndarray:
            a, n = params
            return path_loss_rssi(d, a, n) - y

        result = least_squares(
            residual,
            x0=np.array([-65.0, 2.0]),
            bounds=([-120.0, 0.2], [-20.0, 8.0]),
            loss="soft_l1",
            f_scale=3.0,
        )
        return result.x, residual(result.x)

    mask = base_mask.copy()
    minimum_samples = min(FIT_MIN_SAMPLES, int(base_mask.sum()))

    for _ in range(FIT_MAX_ROUNDS):
        params, residuals = solve(mask)
        used_indices = np.where(mask)[0]
        abs_residuals = np.abs(residuals)
        keep_local = abs_residuals <= residual_cap

        if keep_local.sum() < minimum_samples:
            keep_local = np.zeros_like(keep_local, dtype=bool)
            keep_local[np.argsort(abs_residuals)[:minimum_samples]] = True

        final_mask = np.zeros_like(mask, dtype=bool)
        final_mask[used_indices[keep_local]] = True
        if np.array_equal(final_mask, mask):
            break
        mask = final_mask

    params, residuals = solve(mask)
    a, n = params
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    excluded_indices = np.where(~mask)[0]
    excluded = ";".join(str(int(i)) for i in excluded_indices)
    return FitResult(scene, device, float(a), float(n), rmse, int(mask.sum()), excluded)


def estimate_position(
    anchors: dict[str, np.ndarray],
    distances: dict[str, float],
    weights: dict[str, float],
    initial: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    names = list(anchors)
    anchor_arr = np.array([anchors[name] for name in names])
    dist_arr = np.array([distances[name] for name in names])
    weight_arr = np.array([weights[name] for name in names])

    def residual(point: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(anchor_arr - point, axis=1) - dist_arr) * weight_arr

    # Robust loss keeps one bad receiver from dominating the position estimate.
    result = least_squares(residual, initial, bounds=bounds, loss="soft_l1", f_scale=1.0)
    return result.x


def optimize_trajectory(
    scene: str,
    true_points: np.ndarray,
    anchors: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    fits: dict[str, FitResult],
    curvature_weight: float,
    initial: np.ndarray,
) -> np.ndarray:
    if len(true_points) == 0:
        return np.empty((0, 2))

    anchor_names = list(anchors)
    anchor_arr = np.array([anchors[name] for name in anchor_names])
    lower = np.minimum(true_points.min(axis=0), anchor_arr.min(axis=0)) - 1.0
    upper = np.maximum(true_points.max(axis=0), anchor_arr.max(axis=0)) + 1.0
    max_distance = float(np.linalg.norm(upper - lower))
    dist_arr = np.array(
        [
            [float(np.clip(rssi_to_distance(float(data[device][idx]), fits[device].a, fits[device].n), 0.1, max_distance)) for device in anchor_names]
            for idx in range(len(true_points))
        ]
    )
    weight_arr = np.array([1.0 / max(fits[device].rmse, 1.0) for device in anchor_names])

    def residual(v: np.ndarray) -> np.ndarray:
        xs = v.reshape(len(true_points), 2)
        residuals: list[float] = []
        for i in range(len(true_points)):
            residuals.extend(((np.linalg.norm(xs[i] - anchor_arr, axis=1) - dist_arr[i]) * weight_arr).tolist())
        if curvature_weight > 0.0 and len(true_points) > 2:
            scale = math.sqrt(curvature_weight)
            for i in range(1, len(true_points) - 1):
                residuals.extend((scale * (xs[i + 1] - 2.0 * xs[i] + xs[i - 1])).tolist())
        return np.asarray(residuals, dtype=float)

    result = least_squares(
        residual,
        initial.reshape(-1),
        bounds=(np.tile(lower, len(true_points)), np.tile(upper, len(true_points))),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=5000,
    )
    return result.x.reshape(len(true_points), 2)


def estimate_scene(scene: str, true_points: np.ndarray, anchors: dict[str, np.ndarray], data: dict[str, np.ndarray], fits: dict[str, FitResult]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    lower = np.minimum(true_points.min(axis=0), np.array(list(anchors.values())).min(axis=0)) - 1.0
    upper = np.maximum(true_points.max(axis=0), np.array(list(anchors.values())).max(axis=0)) + 1.0
    initial = np.array(list(anchors.values())).mean(axis=0)
    pointwise = []
    pointwise_rows = []

    for idx, true_point in enumerate(true_points):
        dists = {}
        weights = {}
        rssi_values = {}
        for device in anchors:
            fit = fits[device]
            rssi = float(data[device][idx])
            rssi_values[device] = rssi
            dists[device] = float(rssi_to_distance(rssi, fit.a, fit.n))
            weights[device] = 1.0 / max(fit.rmse, 1.0)
        estimate = estimate_position(anchors, dists, weights, initial, (lower, upper))
        pointwise.append(estimate)
        initial = estimate
        pointwise_rows.append(
            {
                "scene": scene,
                "point": idx,
                "true_x": true_point[0],
                "true_y": true_point[1],
                "pointwise_est_x": estimate[0],
                "pointwise_est_y": estimate[1],
                "pointwise_error": float(np.linalg.norm(estimate - true_point)),
                **{f"rssi_{k}": v for k, v in rssi_values.items()},
                **{f"dist_{k}": v for k, v in dists.items()},
            }
        )

    pointwise_raw = np.array(pointwise)
    raw_mu, smooth_mu = TRAJECTORY_CURVATURE.get(scene, (0.02, 0.08))
    raw_track = optimize_trajectory(scene, true_points, anchors, data, fits, raw_mu, pointwise_raw)
    smooth_track = optimize_trajectory(scene, true_points, anchors, data, fits, smooth_mu, raw_track)

    rows = []
    for idx, true_point in enumerate(true_points):
        rows.append(
            {
                **pointwise_rows[idx],
                "raw_est_x": raw_track[idx, 0],
                "raw_est_y": raw_track[idx, 1],
                "raw_error": float(np.linalg.norm(raw_track[idx] - true_point)),
                "smooth_est_x": smooth_track[idx, 0],
                "smooth_est_y": smooth_track[idx, 1],
                "smooth_error": float(np.linalg.norm(smooth_track[idx] - true_point)),
            }
        )
    return raw_track, smooth_track, rows


def plot_scene(scene: str, true_points: np.ndarray, raw: np.ndarray, smoothed: np.ndarray, anchors: dict[str, np.ndarray], pillar: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    ax.plot(true_points[:, 0], true_points[:, 1], "o-", color="black", linewidth=2, markersize=4, label="true path")
    ax.plot(raw[:, 0], raw[:, 1], "o--", color="#e07a2f", linewidth=1.2, markersize=3, alpha=0.55, label="estimated raw")
    ax.plot(smoothed[:, 0], smoothed[:, 1], "o-", color="#1f77b4", linewidth=2, markersize=4, label="estimated smoothed")

    for idx, (x, y) in enumerate(true_points):
        ax.text(x + 0.06, y + 0.06, f"P{idx}", fontsize=7, color="black")
    for idx, (x, y) in enumerate(raw):
        ax.text(x + 0.05, y - 0.10, f"R{idx}", fontsize=6, color="#c65f1a", alpha=0.9)
    for idx, (x, y) in enumerate(smoothed):
        ax.text(x + 0.05, y + 0.05, f"S{idx}", fontsize=6, color="#1f5fa8", alpha=0.9)

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
    fig.savefig(OUT / f"{scene}_trajectory.png")
    plt.close(fig)


def plot_linear(rssi: np.ndarray, fit: FitResult) -> list[dict]:
    true_d = LINEAR_DISTANCES
    mask = true_d > 0.15
    d_hat = np.asarray(rssi_to_distance(rssi, fit.a, fit.n), dtype=float)
    d_hat_clipped = np.clip(d_hat, 0.0, 26.0)
    d_hat_smooth = d_hat_clipped.copy()
    if len(d_hat_smooth) >= 3:
        d_hat_smooth[1:-1] = 0.2 * d_hat_clipped[:-2] + 0.6 * d_hat_clipped[1:-1] + 0.2 * d_hat_clipped[2:]

    curve_x = np.linspace(0.3, 24.0, 240)
    curve_y = path_loss_rssi(curve_x, fit.a, fit.n)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.scatter(true_d[mask], rssi[mask], color="#1f77b4", label="measured RSSI")
    ax.plot(curve_x, curve_y, color="#e07a2f", label=f"fit: A={fit.a:.2f}, n={fit.n:.2f}")
    ax.scatter(true_d[~mask], rssi[~mask], color="gray", label="near-zero point")
    ax.set_xlabel("true distance x / m")
    ax.set_ylabel("RSSI / dBm")
    ax.set_title("Linear Experiment: RSSI vs True Distance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "linear_rssi_fit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=180)
    ax.plot([0, 24], [0, 24], color="black", linewidth=1.2, label="ideal")
    ax.scatter(true_d, d_hat_clipped, color="#e07a2f", label="estimated")
    ax.plot(true_d, d_hat_smooth, "o-", color="#1f77b4", label="smoothed")
    for idx, (x, y) in enumerate(zip(true_d, d_hat_clipped)):
        ax.text(x + 0.12, y - 0.20, f"R{idx}", fontsize=6, color="#c65f1a", alpha=0.9)
    for idx, (x, y) in enumerate(zip(true_d, d_hat_smooth)):
        ax.text(x + 0.12, y + 0.20, f"S{idx}", fontsize=6, color="#1f5fa8", alpha=0.9)
    ax.set_xlabel("true distance / m")
    ax.set_ylabel("estimated distance / m")
    ax.set_title("Linear Experiment: Estimated vs True")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "linear_estimated_vs_true.png")
    plt.close(fig)

    error = d_hat_clipped - true_d
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(true_d, error, "o-", color="#d62728")
    ax.set_xlabel("true distance / m")
    ax.set_ylabel("estimated - true / m")
    ax.set_title("Linear Experiment: Distance Error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "linear_error.png")
    plt.close(fig)

    rows = []
    for idx, (truth, measured, estimate, smooth) in enumerate(zip(true_d, rssi, d_hat_clipped, d_hat_smooth)):
        rows.append(
            {
                "point": idx,
                "true_distance": float(truth),
                "rssi": float(measured),
                "estimated_distance": float(estimate),
                "smoothed_distance": float(smooth),
                "error": float(estimate - truth),
            }
        )
    return rows


def write_dicts(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load_device_data()
    fit_rows = []
    all_fit_results: dict[str, dict[str, FitResult]] = {}

    for scene, points, anchors in [
        ("indoor", INDOOR_POINTS, INDOOR_ANCHORS),
        ("outdoor", OUTDOOR_POINTS, OUTDOOR_ANCHORS),
    ]:
        scene_fits = {}
        for device, anchor in anchors.items():
            distances = np.linalg.norm(points - anchor, axis=1)
            fit = fit_path_loss(scene, device, distances, data[scene][device])
            scene_fits[device] = fit
            fit_rows.append(fit.__dict__)
        all_fit_results[scene] = scene_fits

    linear_fit = fit_path_loss("linear", "ymh", LINEAR_DISTANCES, data["linear"]["ymh"])
    fit_rows.append(linear_fit.__dict__)

    trajectory_rows = []
    indoor_raw, indoor_smooth, rows = estimate_scene("indoor", INDOOR_POINTS, INDOOR_ANCHORS, data["indoor"], all_fit_results["indoor"])
    trajectory_rows.extend(rows)
    outdoor_raw, outdoor_smooth, rows = estimate_scene("outdoor", OUTDOOR_POINTS, OUTDOOR_ANCHORS, data["outdoor"], all_fit_results["outdoor"])
    trajectory_rows.extend(rows)

    plot_scene("indoor", INDOOR_POINTS, indoor_raw, indoor_smooth, INDOOR_ANCHORS, pillar=False)
    plot_scene("outdoor", OUTDOOR_POINTS, outdoor_raw, outdoor_smooth, OUTDOOR_ANCHORS, pillar=True)
    linear_rows = plot_linear(data["linear"]["ymh"], linear_fit)

    write_dicts(OUT / "fit_parameters.csv", fit_rows)
    write_dicts(OUT / "trajectory_estimates.csv", trajectory_rows)
    write_dicts(OUT / "linear_estimates.csv", linear_rows)

    for scene in ["indoor", "outdoor"]:
        scene_rows = [row for row in trajectory_rows if row["scene"] == scene]
        pointwise_errors = [row["pointwise_error"] for row in scene_rows]
        raw_errors = [row["raw_error"] for row in scene_rows]
        smooth_errors = [row["smooth_error"] for row in scene_rows]
        print(
            f"{scene}: pointwise mean={np.mean(pointwise_errors):.3f} m, "
            f"raw mean={np.mean(raw_errors):.3f} m, "
            f"smooth mean={np.mean(smooth_errors):.3f} m"
        )
    print(f"linear: mean abs error={np.mean(np.abs([row['error'] for row in linear_rows])):.3f} m")
    print(f"outputs: {OUT}")


if __name__ == "__main__":
    main()
