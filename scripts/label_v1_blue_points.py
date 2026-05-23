from __future__ import annotations

import numpy as np

import label_v2_blue_points as base


base.OUT = base.ROOT / "positioning_results_v1_labeled"


def ema_smooth(points: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    smoothed = np.empty_like(points)
    smoothed[0] = points[0]
    for i in range(1, len(points)):
        smoothed[i] = alpha * points[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def main() -> None:
    base.OUT.mkdir(exist_ok=True)
    indoor_raw = base.read_trajectory("indoor")
    outdoor_raw = base.read_trajectory("outdoor")
    base.plot_scene("indoor", base.INDOOR_POINTS, indoor_raw, ema_smooth(indoor_raw), base.INDOOR_ANCHORS)
    base.plot_scene("outdoor", base.OUTDOOR_POINTS, outdoor_raw, ema_smooth(outdoor_raw), base.OUTDOOR_ANCHORS, pillar=True)
    base.plot_linear()
    print(f"labeled v1 plots: {base.OUT}")


if __name__ == "__main__":
    main()
