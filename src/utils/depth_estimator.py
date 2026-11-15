"""
Bathymetric Depth Heatmap and Colorization Module
=================================================
Provides colorization and visual overlay utilities for flood depth maps and
bathymetric water depth models.
"""

from __future__ import annotations

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colormap Definitions
# ---------------------------------------------------------------------------
COLORMAP_DICT = {
    "turbo":   cv2.COLORMAP_TURBO,    # Blue -> Cyan -> Green -> Yellow -> Red (Bathymetric signature)
    "inferno": cv2.COLORMAP_INFERNO,  # Black -> Purple -> Red -> Yellow
    "magma":   cv2.COLORMAP_MAGMA,
    "plasma":  cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def resolve_colormap(cm: str | int) -> int:
    """Resolve colormap name or integer ID to a valid cv2.COLORMAP_* integer."""
    if isinstance(cm, int):
        return cm
    return COLORMAP_DICT.get(str(cm).lower(), cv2.COLORMAP_TURBO)


def colorize_depth_map(
    depth_raw: np.ndarray,
    colormap: str | int = "turbo",
    add_colorbar: bool = True,
    title: str = "BATHYMETRIC FLOOD DEPTH HEATMAP",
) -> np.ndarray:
    """
    Colorize a depth map into a high-contrast visual heatmap.

    Parameters
    ----------
    depth_raw : np.ndarray shape (H, W) float32
        Calibrated water depth (in cm) or relative bathymetry.
    colormap : str or int
        'turbo' (default), 'inferno', 'magma', 'plasma', 'viridis' or cv2 constant.
    add_colorbar : bool
        If True, embeds a HUD scale bar and telemetry legend.
    title : str
        Header banner text.

    Returns
    -------
    heatmap_bgr : np.ndarray shape (H, W, 3) uint8 BGR image
    """
    h, w = depth_raw.shape[:2]
    cm_id = resolve_colormap(colormap)
    cm_name = colormap if isinstance(colormap, str) else "Turbo"

    d_min = float(np.percentile(depth_raw, 1))
    d_max = float(np.percentile(depth_raw, 99))
    if d_max > d_min:
        depth_norm = np.clip((depth_raw - d_min) / (d_max - d_min), 0.0, 1.0)
    else:
        depth_norm = np.zeros_like(depth_raw, dtype=np.float32)

    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_u8, cm_id)

    if not add_colorbar:
        return heatmap

    # Embed HUD overlay with colorbar and title banner
    annotated = heatmap.copy()
    banner_h = max(42, int(h * 0.09))

    # Top dark banner
    top_overlay = annotated[0:banner_h, 0:w].copy()
    black_bar = np.zeros_like(top_overlay)
    annotated[0:banner_h, 0:w] = cv2.addWeighted(top_overlay, 0.25, black_bar, 0.75, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, h / 1000.0)

    # Title & colormap badge
    cv2.putText(
        annotated, title, (14, int(banner_h * 0.65)),
        font, font_scale * 1.05, (255, 255, 255), 1, cv2.LINE_AA
    )
    badge_text = f"Colormap: {str(cm_name).capitalize()}"
    cv2.putText(
        annotated, badge_text, (max(14, w - 180), int(banner_h * 0.65)),
        font, font_scale * 0.85, (200, 230, 255), 1, cv2.LINE_AA
    )

    # Bottom Depth Scale Gradient Bar
    bar_w = min(260, int(w * 0.45))
    bar_h = 10
    bx = 14
    by = h - 22

    # Draw gradient strip
    gradient = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, -1)
    gradient = np.repeat(gradient, bar_h, axis=0)
    gradient_colored = cv2.applyColorMap(gradient, cm_id)

    # Gradient background box
    box_pad = 6
    bg_box = annotated[by - 18:by + bar_h + 16, bx - box_pad:bx + bar_w + box_pad].copy()
    black_bg = np.zeros_like(bg_box)
    annotated[by - 18:by + bar_h + 16, bx - box_pad:bx + bar_w + box_pad] = cv2.addWeighted(
        bg_box, 0.25, black_bg, 0.75, 0
    )

    annotated[by:by + bar_h, bx:bx + bar_w] = gradient_colored
    cv2.rectangle(annotated, (bx, by), (bx + bar_w, by + bar_h), (255, 255, 255), 1)

    # Scale labels: Shallow / Dry (0) -> Deep Water (max)
    cv2.putText(annotated, "SHALLOW / DRY", (bx, by - 4), font, font_scale * 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(annotated, "DEEP WATER", (bx + bar_w - 85, by - 4), font, font_scale * 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    return annotated


def blend_depth_heatmap(
    bgr_img: np.ndarray,
    depth_heatmap: np.ndarray,
    alpha: float = 0.55
) -> np.ndarray:
    """Blend the bathymetric depth heatmap onto the original RGB/BGR image."""
    h, w = bgr_img.shape[:2]
    if depth_heatmap.shape[:2] != (h, w):
        depth_heatmap = cv2.resize(depth_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(bgr_img, 1.0 - alpha, depth_heatmap, alpha, 0)
