from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
except ImportError:  # pragma: no cover - fallback for minimal Python installs
    ndi = None
    peak_local_max = None
    watershed = None


TICKS = np.array([0, -20, -40, -60, -80, -100], dtype=float)


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def grayish_mask(img: np.ndarray) -> np.ndarray:
    r = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    b = img[:, :, 2].astype(int)
    return (np.abs(r - g) < 12) & (np.abs(r - b) < 12) & (r > 80) & (r < 235)


def red_mask(img: np.ndarray) -> np.ndarray:
    r = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    b = img[:, :, 2].astype(int)
    return (r > 180) & (g < 120) & (b < 120) & (r > g + 40) & (r > b + 40)


def yellow_mask(img: np.ndarray) -> np.ndarray:
    r = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    b = img[:, :, 2].astype(int)
    return (r > 220) & (g > 150) & (b < 90) & (r > g) & (g > b + 70)


def blue_mask(img: np.ndarray) -> np.ndarray:
    r = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    b = img[:, :, 2].astype(int)
    return (b > 140) & (g > 80) & (r < 120) & (b > r + 50)


def cluster_consecutive_rows(rows: list[int]) -> list[float]:
    if not rows:
        return []
    groups: list[list[int]] = []
    cur = [rows[0]]
    for y in rows[1:]:
        if y == cur[-1] + 1:
            cur.append(y)
        else:
            groups.append(cur)
            cur = [y]
    groups.append(cur)
    return [sum(g) / len(g) for g in groups]


def find_grid_lines(img: np.ndarray) -> list[float]:
    h, w = img.shape[:2]
    mask = grayish_mask(img)
    y0 = int(h * 0.08)
    y1 = int(h * 0.42)
    x0 = int(w * 0.08)
    x1 = int(w * 0.98)
    row_scores = mask[y0:y1, x0:x1].sum(axis=1)

    # Candidate rows are mostly the horizontal grid lines.
    rows = np.where(row_scores > w * 0.55)[0] + y0
    centers = cluster_consecutive_rows(rows.tolist())
    if len(centers) < 6:
        raise RuntimeError(f"not enough grid lines found: {centers}")

    # Select the best run of 6 nearly equally spaced lines.
    best = None
    for i in range(len(centers) - 5):
        run = centers[i : i + 6]
        diffs = np.diff(run)
        score = float(np.std(diffs)) + abs(float(np.mean(diffs)) - 100.0) / 10.0
        if best is None or score < best[0]:
            best = (score, run)
    assert best is not None
    return list(best[1])


def connected_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        xs = []
        ys = []
        while q:
            cy, cx = q.popleft()
            xs.append(cx)
            ys.append(cy)
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        area = len(xs)
        if area >= 10:
            comps.append(
                {
                    "area": area,
                    "cx": float(sum(xs) / area),
                    "cy": float(sum(ys) / area),
                    "xmin": min(xs),
                    "xmax": max(xs),
                    "ymin": min(ys),
                    "ymax": max(ys),
                    "pixels": np.array(list(zip(xs, ys)), dtype=float),
                }
            )
    return comps


def split_merged_components(comps: list[dict]) -> list[dict]:
    single_areas = [c["area"] for c in comps if c["area"] <= 95]
    typical_area = float(np.median(single_areas)) if single_areas else 68.0
    typical_width = float(np.median([c["xmax"] - c["xmin"] + 1 for c in comps])) if comps else 10.0
    typical_height = float(np.median([c["ymax"] - c["ymin"] + 1 for c in comps])) if comps else 9.0
    split = []
    for comp in comps:
        width = comp["xmax"] - comp["xmin"] + 1
        height = comp["ymax"] - comp["ymin"] + 1
        suspicious = (
            comp["area"] > typical_area * 1.7
            and (width > typical_width * 1.45 or height > typical_height * 1.45)
        )
        if not suspicious:
            split.append(comp)
            continue

        watershed_parts = split_component_by_watershed(comp, typical_area)
        if len(watershed_parts) > 1:
            split.extend(watershed_parts)
            continue

        pixels = comp["pixels"]
        remaining = {(int(x), int(y)) for x, y in pixels}
        centers: list[tuple[float, float, int]] = []
        radius = 5
        score_offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
        search_margin = 2
        first_score = None
        while len(remaining) >= max(12, typical_area * 0.25):
            best = None
            for cy in range(comp["ymin"] - search_margin, comp["ymax"] + search_margin + 1):
                for cx in range(comp["xmin"] - search_margin, comp["xmax"] + search_margin + 1):
                    score = sum((cx + dx, cy + dy) in remaining for dx, dy in score_offsets)
                    if best is None or score > best[0]:
                        best = (score, cx, cy)
            if best is None:
                break
            score, cx, cy = best
            if first_score is None:
                first_score = score
            elif score < max(18, first_score * 0.42):
                break
            if any(math.hypot(cx - ox, cy - oy) < 6.0 for ox, oy, _ in centers):
                break
            centers.append((float(cx), float(cy), int(score)))
            remove_radius = 6
            for dy in range(-remove_radius, remove_radius + 1):
                for dx in range(-remove_radius, remove_radius + 1):
                    if dx * dx + dy * dy <= remove_radius * remove_radius:
                        remaining.discard((cx + dx, cy + dy))

        if len(centers) <= 1:
            split.append(comp)
            continue

        arr = np.array([[x, y] for x, y in pixels], dtype=float)
        center_xy = np.array([[x, y] for x, y, _ in centers], dtype=float)
        labels = np.argmin(
            ((arr[:, None, :] - center_xy[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        for i in range(len(centers)):
            part = arr[labels == i]
            if len(part) < 8:
                continue
            px = part[:, 0]
            py = part[:, 1]
            split.append(
                {
                    "area": int(len(part)),
                    "cx": float(px.mean()),
                    "cy": float(py.mean()),
                    "xmin": int(px.min()),
                    "xmax": int(px.max()),
                    "ymin": int(py.min()),
                    "ymax": int(py.max()),
                }
            )
    return split


def split_component_by_watershed(comp: dict, typical_area: float) -> list[dict]:
    if ndi is None or peak_local_max is None or watershed is None:
        return []

    pixels = comp["pixels"].astype(int)
    pad = 4
    xmin = comp["xmin"]
    xmax = comp["xmax"]
    ymin = comp["ymin"]
    ymax = comp["ymax"]
    local = np.zeros((ymax - ymin + 1 + pad * 2, xmax - xmin + 1 + pad * 2), dtype=bool)
    xs = pixels[:, 0] - xmin + pad
    ys = pixels[:, 1] - ymin + pad
    local[ys, xs] = True

    dist = ndi.distance_transform_edt(local)
    peaks = peak_local_max(
        dist,
        labels=local,
        min_distance=5,
        threshold_abs=2.5,
        exclude_border=False,
    )
    if len(peaks) <= 1:
        return []

    max_expected = max(2, int(round(comp["area"] / typical_area)) + 1)
    if len(peaks) > max_expected:
        scores = [dist[y, x] for y, x in peaks]
        keep = np.argsort(scores)[-max_expected:]
        peaks = peaks[keep]

    markers = np.zeros_like(local, dtype=int)
    for i, (py, px) in enumerate(peaks, start=1):
        markers[py, px] = i

    labels = watershed(-dist, markers, mask=local)
    parts = []
    for label_id in range(1, labels.max() + 1):
        py, px = np.where(labels == label_id)
        if len(px) < 8:
            continue
        orig_x = px + xmin - pad
        orig_y = py + ymin - pad
        parts.append(
            {
                "area": int(len(px)),
                "cx": float(orig_x.mean()),
                "cy": float(orig_y.mean()),
                "xmin": int(orig_x.min()),
                "xmax": int(orig_x.max()),
                "ymin": int(orig_y.min()),
                "ymax": int(orig_y.max()),
            }
        )
    return parts if len(parts) > 1 else []


def estimate_rssi(cy: float, grid: list[float]) -> float:
    if cy <= grid[0]:
        return float(TICKS[0])
    if cy >= grid[-1]:
        return float(TICKS[-1])
    for i in range(len(grid) - 1):
        y0, y1 = grid[i], grid[i + 1]
        if y0 <= cy <= y1:
            t = (cy - y0) / (y1 - y0) if y1 != y0 else 0.0
            return float((1 - t) * TICKS[i] + t * TICKS[i + 1])
    return float("nan")


def read_android_image(path: Path) -> dict:
    img = load_image(path)
    grid = find_grid_lines(img)
    y_min = max(0, int(grid[0] - 6))
    y_max = min(img.shape[0] - 1, int(grid[-1] + 6))
    mask = red_mask(img)
    region = mask[y_min : y_max + 1, :]
    comps = split_merged_components(connected_components(region))
    comps.sort(key=lambda item: item["cx"])
    points = []
    for comp in comps:
        area = comp["area"]
        cx = comp["cx"]
        cy = comp["cy"]
        rssi = estimate_rssi(cy + y_min, grid)
        points.append(
            {
                "x": round(cx, 1),
                "y": round(cy + y_min, 1),
                "rssi": round(rssi),
                "rssi_float": round(rssi, 2),
                "area": area,
            }
        )
    return {"file": path.name, "grid": [round(v, 1) for v in grid], "points": points}


def find_ios_chart_bounds(img: np.ndarray) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    mask = grayish_mask(img)
    y0 = int(h * 0.18)
    y1 = int(h * 0.62)
    x0 = int(w * 0.12)
    x1 = int(w * 0.96)
    region = mask[y0:y1, x0:x1]
    row_scores = region.sum(axis=1)
    col_scores = region.sum(axis=0)

    rows = np.where(row_scores > 250)[0] + y0
    cols = np.where(col_scores > 150)[0] + x0
    row_centers = cluster_consecutive_rows(rows.tolist())
    col_centers = cluster_consecutive_rows(cols.tolist())
    horizontal = [r for r in row_centers if y0 + 50 < r < y1 - 20]
    vertical = [c for c in col_centers if x0 + 40 < c < x1 - 20]
    if len(horizontal) < 3 or len(vertical) < 2:
        raise RuntimeError(f"cannot find iOS chart bounds: rows={row_centers}, cols={col_centers}")
    return (
        int(min(vertical) - 40),
        int(max(vertical) + 40),
        int(min(horizontal) - 40),
        int(max(horizontal) + 40),
    )


def read_ios_image(path: Path, min_db: float, max_db: float) -> dict:
    img = load_image(path)
    x_min, x_max, y_min, y_max = find_ios_chart_bounds(img)
    mask_y = yellow_mask(img)
    mask_b = blue_mask(img)

    region_y = mask_y[y_min : y_max + 1, x_min : x_max + 1]
    region_b = mask_b[y_min : y_max + 1, x_min : x_max + 1]
    comps_y = split_merged_components(connected_components(region_y))
    comps_b = split_merged_components(connected_components(region_b))
    comps_y.sort(key=lambda item: item["cx"])
    comps_b.sort(key=lambda item: item["cx"])
    if len(comps_y) < 2 and len(comps_b) < 2:
        raise RuntimeError(f"not enough points found in {path}")

    all_comps = [(c, "yellow") for c in comps_y] + [(c, "blue") for c in comps_b]
    all_comps.sort(key=lambda t: t[0]["cx"])
    ys = [c["cy"] for c, _ in all_comps]
    top_y = min(ys)
    bottom_y = max(ys)
    if abs(bottom_y - top_y) < 1:
        bottom_y = top_y + 1.0

    points = []
    for comp, series in all_comps:
        cx = comp["cx"] + x_min
        cy = comp["cy"] + y_min
        t = (comp["cy"] - top_y) / (bottom_y - top_y)
        rssi = max_db + t * (min_db - max_db)
        points.append(
            {
                "series": series,
                "x": round(cx, 1),
                "y": round(cy, 1),
                "rssi": round(rssi),
                "rssi_float": round(rssi, 2),
                "area": comp["area"],
            }
        )
    return {
        "file": path.name,
        "mode": "ios",
        "min_db": min_db,
        "max_db": max_db,
        "chart_bounds": [x_min, x_max, y_min, y_max],
        "points": points,
    }


def read_ranges_csv(path: Path) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file") or row.get("filename") or row.get("name")
            if not file_name:
                continue
            min_value = row.get("min_db") or row.get("min") or row.get("Min")
            max_value = row.get("max_db") or row.get("max") or row.get("Max")
            if min_value is None or max_value is None:
                continue
            ranges[file_name] = (float(min_value), float(max_value))
    return ranges


def annotate_image(src: Path, result: dict, out: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    for idx, point in enumerate(result["points"], start=1):
        x = float(point["x"])
        y = float(point["y"])
        rssi = point["rssi"]
        color = (0, 120, 255) if point.get("series") == "blue" else (255, 170, 0)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=2)
        draw.text((x + 6, y - 13), f"{idx}:{rssi}", fill=color)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def write_csv(results: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "point_index", "series", "x_px", "y_px", "rssi", "rssi_float", "area"])
        for item in results:
            for idx, point in enumerate(item["points"], start=1):
                writer.writerow(
                    [
                        item["file"],
                        idx,
                        point.get("series", ""),
                        point["x"],
                        point["y"],
                        point["rssi"],
                        point["rssi_float"],
                        point["area"],
                    ]
                )


def iqr_filtered(values: np.ndarray) -> np.ndarray:
    if len(values) < 4:
        return values
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    filtered = values[(values >= low) & (values <= high)]
    return filtered if len(filtered) else values


def write_summary_csv(results: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "series",
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "iqr_filtered_count",
                "iqr_filtered_mean",
                "iqr_filtered_median",
            ]
        )
        for item in results:
            series_names = sorted({p.get("series", "all") or "all" for p in item["points"]})
            if "all" not in series_names:
                series_names = ["all"] + series_names
            for series_name in series_names:
                if series_name == "all":
                    selected = item["points"]
                else:
                    selected = [p for p in item["points"] if p.get("series") == series_name]
                values = np.array([p["rssi"] for p in selected], dtype=float)
                filtered = iqr_filtered(values)
                writer.writerow(
                    [
                        item["file"],
                        series_name,
                        len(values),
                        round(float(values.mean()), 3) if len(values) else "",
                        round(float(np.median(values)), 3) if len(values) else "",
                        round(float(values.std(ddof=1)), 3) if len(values) > 1 else "",
                        int(values.min()) if len(values) else "",
                        int(values.max()) if len(values) else "",
                        len(filtered),
                        round(float(filtered.mean()), 3) if len(filtered) else "",
                        round(float(np.median(filtered)), 3) if len(filtered) else "",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read RSSI red dots from Lab2 screenshots.")
    parser.add_argument("paths", nargs="*", help="Image files or folders. Default: current folder.")
    parser.add_argument("--mode", choices=["android", "ios"], default="android")
    parser.add_argument("--ranges-csv", help="CSV with columns file,min_db,max_db for iOS screenshots.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--annotate-dir", help="Write images with numbered detected points.")
    parser.add_argument("--csv", help="Write detected points to CSV.")
    parser.add_argument("--summary-csv", help="Write per-image RSSI summary statistics to CSV.")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.paths] if args.paths else [Path(".")]
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            image_files = [
                f for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            files.extend(sorted(image_files, key=lambda x: (int(x.stem) if x.stem.isdigit() else 10**9, x.name)))
        elif p.is_file():
            files.append(p)
    if not files:
        raise SystemExit("No image files found.")

    ranges = read_ranges_csv(Path(args.ranges_csv)) if args.ranges_csv else {}
    results = []
    for f in files:
        if args.mode == "ios":
            if f.name not in ranges:
                raise SystemExit(f"Missing min/max range for {f.name}")
            min_db, max_db = ranges[f.name]
            results.append(read_ios_image(f, min_db, max_db))
        else:
            results.append(read_android_image(f))
    if args.annotate_dir:
        out_dir = Path(args.annotate_dir)
        for src, result in zip(files, results):
            annotate_image(src, result, out_dir / src.name)
    if args.csv:
        write_csv(results, Path(args.csv))
    if args.summary_csv:
        write_summary_csv(results, Path(args.summary_csv))
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for item in results:
        values = [p["rssi"] for p in item["points"]]
        print(f'{item["file"]}: {values}')


if __name__ == "__main__":
    main()
