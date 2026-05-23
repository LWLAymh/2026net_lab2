# positioning versions

## v1
- Location: `positioning_results/`
- Method: point-by-point RSSI -> distance conversion, then trilateration for each point.
- Calibration: path-loss fit per device/scene with manual bad-point exclusion and residual pruning.
- Smoothing: simple exponential moving average on the trajectory.
- Output style: raw + smoothed trajectory, plus linear RSSI/distance plots.

## v2
- Location: `positioning_results_v2/`
- Method: same point-by-point trilateration style as v1.
- Calibration: same cleaned path-loss fitting logic, still per device/scene.
- Smoothing: centered 3-point weighted average, `0.2 / 0.6 / 0.2`.
- Output style: raw + smoothed trajectory, with the smoother intended only for visualization.

## v3
- Location: `positioning_results_v3/`
- Method: keep the same path-loss calibration, but replace pointwise-only trajectory with a global trajectory solve.
- Raw track: global least-squares with a light curvature penalty.
- Smoothed track: same global solve with a stronger curvature penalty.
- Goal: make the raw path closer to the real motion while keeping turns more natural than simple averaging.

## v4
- Location: `positioning_results_v4/`
- Method: same math as v3.
- Change: add point labels on the blue/orange trajectory points.
- Purpose: easier reading and point-to-point checking without changing the fitted result.

## Notes
- `raw` is the main estimation result.
- `smoothed` is only for visualization and should not replace the raw analysis unless explicitly stated.
