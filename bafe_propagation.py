"""
Box-Aligned Feature Extraction (BAFE) for compressed-domain bbox propagation.

Samples motion vectors and DCT energy on a fixed grid aligned to each bounding
box (interior + neighborhood), aggregates per-cell features, and uses robust
median motion to update box position on P-frames.
"""

import numpy as np


def mb_to_grid(mb_x, mb_y, num_mb_x, num_mb_y, grid_size):
    return (
        (mb_x + 0.5) * grid_size / num_mb_x,
        (mb_y + 0.5) * grid_size / num_mb_y,
    )


def mv_to_grid_delta(dx, dy, frame_width, frame_height, grid_size):
    """Convert H.264 quarter-pixel MVs to grid-space displacement.
    H.264 motion vectors point backwards to the reference frame block (X_ref - X_curr),
    so object forward motion direction is (-dx, -dy).
    """
    grid_dx = -(dx / 4.0) * grid_size / frame_width
    grid_dy = -(dy / 4.0) * grid_size / frame_height
    return grid_dx, grid_dy


def extract_bafe_features(
    box,
    mbs,
    num_mb_x,
    num_mb_y,
    grid_size,
    n_cells=3,
    neighborhood_scale=0.2,
):
    """
    Extract box-aligned MV and DCT features on an n_cells x n_cells grid
    covering the bbox interior and its immediate neighborhood.
    """
    _, cx, cy, w, h = box

    margin_x = w * neighborhood_scale
    margin_y = h * neighborhood_scale
    x_min = cx - w / 2 - margin_x
    x_max = cx + w / 2 + margin_x
    y_min = cy - h / 2 - margin_y
    y_max = cy + h / 2 + margin_y

    region_w = x_max - x_min
    region_h = y_max - y_min
    if region_w <= 0 or region_h <= 0:
        return [], []

    cell_w = region_w / n_cells
    cell_h = region_h / n_cells

    cells = [[{"dx": [], "dy": [], "dct_energy": []} for _ in range(n_cells)]
             for _ in range(n_cells)]

    for mb in mbs:
        mb_gx, mb_gy = mb_to_grid(
            mb["mb_x"], mb["mb_y"], num_mb_x, num_mb_y, grid_size
        )

        if not (x_min <= mb_gx <= x_max and y_min <= mb_gy <= y_max):
            continue

        ci = min(int((mb_gx - x_min) / cell_w), n_cells - 1)
        cj = min(int((mb_gy - y_min) / cell_h), n_cells - 1)

        cells[cj][ci]["dx"].append(mb["dx"])
        cells[cj][ci]["dy"].append(mb["dy"])
        cells[cj][ci]["dct_energy"].append(mb["dct_energy"])

    mv_features = []
    dct_features = []

    for row in cells:
        for cell in row:
            if cell["dx"]:
                mv_features.append(
                    (float(np.median(cell["dx"])), float(np.median(cell["dy"])))
                )
                dct_features.append(float(np.median(cell["dct_energy"])))
            else:
                mv_features.append((0.0, 0.0))
                dct_features.append(0.0)

    return mv_features, dct_features


def _mbs_inside_box(box, mbs, num_mb_x, num_mb_y, grid_size):
    """Collect MVs from macroblocks whose center falls inside the bbox."""
    _, cx, cy, w, h = box
    dx_vals = []
    dy_vals = []

    for mb in mbs:
        mb_gx, mb_gy = mb_to_grid(
            mb["mb_x"], mb["mb_y"], num_mb_x, num_mb_y, grid_size
        )

        if (
            (cx - w / 2) <= mb_gx <= (cx + w / 2)
            and (cy - h / 2) <= mb_gy <= (cy + h / 2)
        ):
            dx_vals.append(mb["dx"])
            dy_vals.append(mb["dy"])

    return dx_vals, dy_vals


def propagate_box_bafe(
    box,
    mbs,
    num_mb_x,
    num_mb_y,
    grid_size,
    frame_width,
    frame_height,
    n_cells=3,
    neighborhood_scale=0.2,
):
    """
    Propagate a single bbox on a P-frame using BAFE grid-aligned MV aggregation.

    Returns updated box: [confidence, cx, cy, w, h]
    """
    conf, cx, cy, w, h = box

    mv_features, _ = extract_bafe_features(
        box,
        mbs,
        num_mb_x,
        num_mb_y,
        grid_size,
        n_cells=n_cells,
        neighborhood_scale=neighborhood_scale,
    )

    dx_vals = [mv[0] for mv in mv_features if mv[0] != 0 or mv[1] != 0]
    dy_vals = [mv[1] for mv in mv_features if mv[0] != 0 or mv[1] != 0]

    if not dx_vals:
        dx_vals, dy_vals = _mbs_inside_box(
            box, mbs, num_mb_x, num_mb_y, grid_size
        )

    if not dx_vals:
        return box

    median_dx = float(np.median(dx_vals))
    median_dy = float(np.median(dy_vals))

    grid_dx, grid_dy = mv_to_grid_delta(
        median_dx, median_dy, frame_width, frame_height, grid_size
    )

    new_cx = float(np.clip(cx + grid_dx, w / 2, grid_size - w / 2))
    new_cy = float(np.clip(cy + grid_dy, h / 2, grid_size - h / 2))

    return [conf, new_cx, new_cy, w, h]


def propagate_boxes_bafe(
    active_boxes,
    mbs,
    num_mb_x,
    num_mb_y,
    grid_size,
    frame_width,
    frame_height,
):
    """Propagate all active boxes on a P-frame."""
    return [
        propagate_box_bafe(
            box,
            mbs,
            num_mb_x,
            num_mb_y,
            grid_size,
            frame_width,
            frame_height,
        )
        for box in active_boxes
    ]
def filter_roi_macroblocks(active_boxes, mbs, num_mb_x, num_mb_y, grid_size):
    """Keep macroblocks whose center falls inside any propagated bbox."""
    roi_mbs = []

    for mb in mbs:
        mb_gx, mb_gy = mb_to_grid(
            mb["mb_x"], mb["mb_y"], num_mb_x, num_mb_y, grid_size
        )
        for box in active_boxes:
            _, cx, cy, w, h = box

            if (
                (cx - w / 2) <= mb_gx <= (cx + w / 2)
                and (cy - h / 2) <= mb_gy <= (cy + h / 2)
            ):
                roi_mbs.append({
                    "mb_x": mb["mb_x"],
                    "mb_y": mb["mb_y"],
                    "dx": mb["dx"],
                    "dy": mb["dy"],
                    "dct_energy": mb["dct_energy"],
                })
                break

    return roi_mbs
