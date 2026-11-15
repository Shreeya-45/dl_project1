"""
Road Passability Analyzer
=========================
Classifies flooded roads as Passable / Partially Flooded / Severely Affected
using two depth-estimation backends:

1. **STURM-FloodDepth (calibrated physical depth, Zenodo: 14833532)**
   – 5-class vehicle-grounded physical levels (Level 0 to 4).

2. **Morphological (always available)**
   – Euclidean Distance Transform on the flood segmentation mask.
"""

import cv2
import numpy as np

class PassabilityAnalyzer:
    """
    Road Passability and Morphological Flood Depth Analysis Engine.
    Estimates road passability using Flooded-Road Pixel Ratio, Connected Road Geometry,
    and Euclidean Distance Transform (EDT) bathymetric flood depth approximation.
    Classifies scenes into: Passable, Partially Flooded, Severely Affected.
    """
    def __init__(
        self,
        passable_thresh: float = 0.15,
        severe_thresh: float = 0.50,
        max_allowable_depth_cm: float = 50.0,
        sturm_estimator=None,
    ):
        self.passable_thresh = passable_thresh
        self.severe_thresh = severe_thresh
        self.max_allowable_depth_cm = max_allowable_depth_cm
        self._sturm_estimator = sturm_estimator

    def compute_depth_map(self, pred_mask, img_shape):
        """
        Calculates physically calibrated water depth (in cm) from the 2D water mask (pred_mask == 2).
        Returns:
            depth_cm: 2D float array with depth in cm for each pixel.
            max_depth: peak depth in cm.
            mean_depth: average depth of flooded pixels in cm.
        """
        h, w = img_shape[:2]
        water_binary = (pred_mask == 2).astype(np.uint8)

        if np.sum(water_binary) == 0:
            return np.zeros((h, w), dtype=np.float32), 0.0, 0.0

        # Perspective distance Z(u, v) in meters (1.5m at bumper to ~26m at horizon)
        y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)
        norm_y = (h - 1.0 - y_coords) / float(max(1, h))
        z_metric = 1.5 + 25.0 * (norm_y ** 1.4)

        fx = 0.9 * float(w)
        pixel_scale_m = z_metric / fx

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(water_binary, connectivity=8)
        depth_cm = np.zeros((h, w), dtype=np.float32)

        for label in range(1, num_labels):
            pool_mask = (labels == label).astype(np.uint8)
            if np.sum(pool_mask) < 4:
                continue

            dist_px = cv2.distanceTransform(pool_mask, cv2.DIST_L2, 5)
            d_max_px = float(dist_px.max())
            if d_max_px <= 0:
                continue

            r_world_m = dist_px * pixel_scale_m
            r_peak_m = float(np.max(r_world_m[pool_mask > 0]))

            peak_depth_cm = float(np.clip(15.0 * (r_peak_m ** 0.72), 0.8, self.max_allowable_depth_cm))
            norm_dist = (dist_px / d_max_px) ** 0.68
            pool_depth = norm_dist * peak_depth_cm
            depth_cm[pool_mask > 0] = pool_depth[pool_mask > 0]

        max_depth = float(depth_cm.max())
        water_pixels = depth_cm[water_binary == 1]
        mean_depth = float(np.mean(water_pixels)) if len(water_pixels) > 0 else 0.0

        return depth_cm, round(max_depth, 1), round(mean_depth, 1)

    def analyze(self, pred_mask, colormap="turbo", generate_heatmap: bool = True):
        """
        Runs passability analysis using the morphological distance depth backend.

        pred_mask: 2D numpy array with class indices (0: Background, 1: Dry Road, 2: Water/Flood)
        Returns dictionary containing passability status, ratios, depth telemetry, and connected geometry.
        """
        h, w = pred_mask.shape[:2]

        # Fast pixel counting via bincount (single pass, 3× faster than 3 np.sum calls)
        flat = pred_mask.ravel()
        counts = np.bincount(flat, minlength=3)
        road_pixels  = int(counts[1])
        flood_pixels = int(counts[2])

        # Check if explicit dry road class is segmented in the model output
        if road_pixels > 0:
            total_road_surface = road_pixels + flood_pixels
            flooded_ratio = float(flood_pixels / total_road_surface)
            dry_road_pixels = road_pixels
        else:
            # In datasets like CVFD that only annotate Water vs Background,
            # non-water ground is part of Class 0.
            # In close-view ground/street perspective, the roadway corridor is the lower ~70% of the scene.
            ground_corridor_pixels = int(h * w * 0.70)
            total_road_surface = max(flood_pixels, ground_corridor_pixels)
            flooded_ratio = float(flood_pixels / total_road_surface) if total_road_surface > 0 else 0.0
            dry_road_pixels = max(0, total_road_surface - flood_pixels)

        # Morphological Flood Depth Estimation
        depth_cm, max_depth, mean_depth = self.compute_depth_map(pred_mask, (h, w))

        # Connected Component Geometry Analysis on Dry Road Surface
        road_binary = (pred_mask == 1).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(road_binary, connectivity=8)

        # Filter tiny noise components
        min_area = 50
        valid_components = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]
        num_road_segments = len(valid_components)
        road_fragmented = num_road_segments > 2 and flooded_ratio > 0.10

        # Pure Depth-Based Passability Classification:
        # Passability classified strictly using vertical water depth (road/flood pixel ratio removed)
        if max_depth >= 30.0:
            status = "Severely Affected"
            color_bgr = (0, 0, 235)     # Deep Red
            color_hex = "#EF4444"
            recommendation = f"UNSAFE: High water depth ({max_depth} cm). Engine flood hazard. Reroute traffic!"
            badge_class = "severely-affected"
            primary_factor = f"Critical Water Depth ({max_depth} cm >= 30 cm)"
        elif max_depth >= 15.0:
            status = "Partially Flooded"
            color_bgr = (0, 165, 255)   # Orange / Amber
            color_hex = "#F59E0B"
            recommendation = f"CAUTION: Water depth {max_depth} cm. Clearance required (SUVs / Emergency Trucks only). Sedans avoid."
            badge_class = "partially-flooded"
            primary_factor = f"Moderate Water Depth ({max_depth} cm >= 15 cm)"
        else:
            status = "Passable"
            color_bgr = (46, 204, 113)  # Green
            color_hex = "#10B981"
            badge_class = "passable"
            if max_depth <= 2.0:
                recommendation = "Road is Dry / Clear. Passable with normal driving caution."
                primary_factor = "Dry Road / Negligible Water Depth"
            else:
                recommendation = f"Safe to drive with normal caution. Minor shallow water ({max_depth} cm)."
                primary_factor = f"Safe Shallow Depth ({max_depth} cm < 15 cm)"

        res = {
            'status': status,
            'flooded_road_ratio': round(flooded_ratio * 100, 2),
            'max_depth_cm': max_depth,
            'mean_depth_cm': mean_depth,
            'flooded_pixels': int(flood_pixels),
            'dry_road_pixels': int(dry_road_pixels),
            'total_road_pixels': int(total_road_surface),
            'road_segments_count': num_road_segments,
            'is_fragmented': road_fragmented,
            'primary_factor': primary_factor,
            'stream1_spatial': {
                'flooded_ratio_pct': round(flooded_ratio * 100, 2),
                'road_continuity': 'Fragmented' if road_fragmented else 'Continuous Corridor',
                'active_segments': num_road_segments,
            },
            'stream2_depth': {
                'backend': 'Morphological Bathymetry',
                'peak_depth_cm': max_depth,
                'mean_depth_cm': mean_depth,
                'hazard_level': 'Severe' if max_depth >= 30.0 else ('Moderate' if max_depth >= 15.0 else 'Safe'),
            },
            'color_bgr': color_bgr,
            'color_hex': color_hex,
            'recommendation': recommendation,
            'badge_class': badge_class,
            'depth_source': 'morphological',
            '_depth_cm_map': depth_cm,
        }
        if generate_heatmap:
            res['_depth_heatmap_bgr'] = self.generate_depth_heatmap(pred_mask, res, colormap=colormap)
        else:
            res['_depth_heatmap_bgr'] = None
        return res

    # ------------------------------------------------------------------
    # STURM-FloodDepth — Vehicle-grounded physical flood depth analysis
    # ------------------------------------------------------------------
    def analyze_with_sturm(
        self,
        pred_mask: np.ndarray,
        bgr_img: np.ndarray,
        sturm_estimator=None,
        colormap: str | int = "turbo",
        generate_heatmap: bool = True,
        sturm_telemetry: dict | None = None,
    ) -> dict:
        """
        Passability and flood depth analysis using the STURM-FloodDepth model
        (Zenodo DOI: 10.5281/zenodo.14833532, 5-class vehicle-calibrated flood depth levels).

        Provides direct, vehicle-grounded physical depth classes:
          - Level 0: Dry (0 cm) -> Passable
          - Level 1: Sub-axle (<15 cm) -> Passable
          - Level 2: Wheel hub (15-30 cm) -> Partially Flooded (SUVs only)
          - Level 3: Chassis/Door (30-60 cm) -> Severely Affected (Impassable)
          - Level 4: Submerged (>60 cm) -> Severely Affected (Catastrophic)

        Parameters
        ----------
        pred_mask       : 2D numpy array — segmentation mask (0 BG, 1 Road, 2 Water)
        bgr_img         : Original BGR image (OpenCV)
        sturm_estimator : STURMFloodDepthEstimator instance (optional)
        colormap        : Colormap name for depth heatmap

        Returns
        -------
        dict containing passability status, STURM level, calibrated depth, and telemetry.
        """
        # Reuse pre-computed STURM telemetry when available (video caching)
        if sturm_telemetry is None:
            estimator = sturm_estimator or self._sturm_estimator
            if estimator is None:
                try:
                    from .sturm_estimator import get_sturm_estimator
                    estimator = get_sturm_estimator()
                except ImportError:
                    estimator = None

            if estimator is None:
                return self.analyze(pred_mask, colormap=colormap)

            if bgr_img is None:
                return self.analyze(pred_mask, colormap=colormap)

            sturm_telemetry = estimator.estimate(bgr_img)
        pred_level = sturm_telemetry['level']

        h, w = pred_mask.shape[:2]

        # ---- Fast pixel counting using np.bincount (avoid 3 np.sum calls) ----
        flat = pred_mask.ravel()
        counts = np.bincount(flat, minlength=3)
        road_pixels  = int(counts[1])
        flood_pixels = int(counts[2])

        if road_pixels > 0:
            total_road_surface = road_pixels + flood_pixels
            flooded_ratio = float(flood_pixels / total_road_surface)
            dry_road_pixels = road_pixels
        else:
            ground_corridor_pixels = int(h * w * 0.70)
            total_road_surface = max(flood_pixels, ground_corridor_pixels)
            flooded_ratio = float(flood_pixels / total_road_surface) if total_road_surface > 0 else 0.0
            dry_road_pixels = max(0, total_road_surface - flood_pixels)

        # Connected road geometry — downscale to 256×256 for speed
        _h, _w = pred_mask.shape[:2]
        if _h > 256 or _w > 256:
            small_mask = cv2.resize(pred_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        else:
            small_mask = pred_mask
        road_binary = (small_mask == 1).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(road_binary, connectivity=8)
        min_area = 20  # scaled down from 50 for 256×256
        valid_components = [
            stats[i, cv2.CC_STAT_AREA]
            for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] > min_area
        ]
        num_road_segments = len(valid_components)
        road_fragmented = num_road_segments > 2 and flooded_ratio > 0.10

        # Physical Bathymetric Depth Map anchored to STURM predicted depth
        peak_depth_cm = float(sturm_telemetry['expected_depth_cm'])
        if peak_depth_cm <= 0 and pred_level == 0:
            peak_depth_cm = 0.0

        water_binary = (small_mask == 2).astype(np.uint8)
        depth_cm = np.zeros((h, w), dtype=np.float32)

        if np.any(water_binary) and peak_depth_cm > 0:
            dist_px = cv2.distanceTransform(water_binary, cv2.DIST_L2, 5)
            d_max = float(dist_px.max())
            if d_max > 0:
                small_depth = ((dist_px / d_max) ** 0.65) * peak_depth_cm
                small_depth[water_binary == 0] = 0.0
                # Resize depth map back to original resolution
                if small_mask is not pred_mask:
                    depth_cm = cv2.resize(small_depth, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    depth_cm = small_depth

        has_water = np.any(water_binary)
        mean_depth_cm = float(np.mean(depth_cm[depth_cm > 0])) if has_water and np.any(depth_cm > 0) else 0.0
        max_depth_cm = float(depth_cm.max()) if has_water else peak_depth_cm

        # Status and passability determination from STURM Level
        status = sturm_telemetry['passability']
        color_bgr = sturm_telemetry['color_bgr']
        color_hex = sturm_telemetry['color_hex']
        badge_class = sturm_telemetry['badge_class']
        recommendation = sturm_telemetry['recommendation']
        primary_factor = f"STURM {sturm_telemetry['level_label']} ({sturm_telemetry['landmark']})"

        # If segmentation detected substantial water coverage (>50%) but STURM predicted Level 0/1,
        # escalate caution for safety:
        if flooded_ratio >= 0.50 and pred_level <= 1:
            status = "Partially Flooded"
            color_bgr = (0, 165, 255)
            color_hex = "#F59E0B"
            badge_class = "partially-flooded"
            recommendation = "CAUTION: Extensive surface water detected across road (>50% road coverage). Proceed with caution."
            primary_factor = f"Extensive Water Surface ({round(flooded_ratio*100, 1)}% road flooded)"

        res = {
            'status': status,
            'flooded_road_ratio': round(flooded_ratio * 100, 2),
            'max_depth_cm': round(max_depth_cm, 1),
            'mean_depth_cm': round(mean_depth_cm, 1),
            'flooded_pixels': int(flood_pixels),
            'dry_road_pixels': int(dry_road_pixels),
            'total_road_pixels': int(total_road_surface),
            'road_segments_count': num_road_segments,
            'is_fragmented': road_fragmented,
            'primary_factor': primary_factor,
            'stream1_spatial': {
                'flooded_ratio_pct': round(flooded_ratio * 100, 2),
                'road_continuity': 'Fragmented' if road_fragmented else 'Continuous Corridor',
                'active_segments': num_road_segments,
            },
            'stream2_depth': {
                'backend': f"STURM-FloodDepth ({sturm_telemetry['level_label']})",
                'peak_depth_cm': round(max_depth_cm, 1),
                'mean_depth_cm': round(mean_depth_cm, 1),
                'hazard_level': sturm_telemetry['hazard'],
            },
            'color_bgr': color_bgr,
            'color_hex': color_hex,
            'recommendation': recommendation,
            'badge_class': badge_class,
            'depth_source': 'sturm_flooddepth',
            'sturm_level': pred_level,
            'sturm_level_label': sturm_telemetry['level_label'],
            'sturm_level_title': sturm_telemetry['level_title'],
            'sturm_confidence': sturm_telemetry['confidence'],
            'sturm_confidence_pct': sturm_telemetry['confidence_pct'],
            'sturm_depth_range_str': sturm_telemetry['depth_range_str'],
            'sturm_probabilities': sturm_telemetry['probabilities'],
            'sturm_landmark': sturm_telemetry['landmark'],
            'sturm_hazard': sturm_telemetry['hazard'],
            '_depth_cm_map': depth_cm,
            '_sturm_telemetry': sturm_telemetry,
        }
        if generate_heatmap:
            res['_depth_heatmap_bgr'] = self.generate_depth_heatmap(
                pred_mask, res, original_bgr=bgr_img, colormap=colormap
            )
        else:
            res['_depth_heatmap_bgr'] = None
        return res



    def generate_depth_heatmap(
        self,
        pred_mask: np.ndarray,
        analysis_res: dict,
        original_bgr: np.ndarray | None = None,
        colormap: str | int = "turbo",
        highlight_water: bool = False,
    ) -> np.ndarray:
        """
        Generates a clean visual bathymetric depth heatmap for the scene.

        Parameters
        ----------
        pred_mask : np.ndarray (H, W)
            Predicted segmentation mask.
        analysis_res : dict
            Result dictionary from `analyze` or `analyze_with_sturm`.
        original_bgr : np.ndarray | None
            Original BGR scene.
        colormap : str | int
            Colormap name ('turbo', 'inferno', 'magma', etc.) or OpenCV constant.
        highlight_water : bool, default False
            Whether to outline the segmented flood pool.

        Returns
        -------
        heatmap_bgr : np.ndarray (H, W, 3) BGR image
        """
        from .depth_estimator import colorize_depth_map, resolve_colormap

        h, w = pred_mask.shape[:2]
        depth_src = analysis_res.get('depth_source', 'sturm_flooddepth')

        # 1. STURM-FloodDepth Bathymetric Heatmap with Level Card
        if depth_src == 'sturm_flooddepth':
            depth_cm = analysis_res.get('_depth_cm_map')
            if depth_cm is None:
                depth_cm, _, _ = self.compute_depth_map(pred_mask, (h, w))

            max_d = max(1.0, float(analysis_res.get('max_depth_cm', depth_cm.max())))
            norm_depth = np.clip((depth_cm / max_d) * 255.0, 0, 255).astype(np.uint8)
            cm_id = resolve_colormap(colormap)
            heatmap = cv2.applyColorMap(norm_depth, cm_id)
            heatmap[pred_mask != 2] = [15, 23, 42]

            # Embed STURM Level Card HUD in top right
            sturm_meta = analysis_res.get('_sturm_telemetry')
            if sturm_meta:
                try:
                    from .sturm_estimator import get_sturm_estimator
                    est = get_sturm_estimator()
                    card_w = min(360, int(w * 0.46))
                    card_h = max(110, int(h * 0.22))
                    card = est.generate_level_card(sturm_meta, width=card_w, height=card_h)
                    ch, cw = card.shape[:2]
                    if w > cw + 15 and h > ch + 15:
                        heatmap[10:10+ch, w - cw - 10:w - 10] = card
                except Exception:
                    self._draw_interpretation_sign(heatmap, analysis_res)
            else:
                self._draw_interpretation_sign(heatmap, analysis_res)
            return heatmap

        # 2. Fallback to morphological depth map
        depth_cm = analysis_res.get('_depth_cm_map')
        if depth_cm is None:
            depth_cm, _, _ = self.compute_depth_map(pred_mask, (h, w))

        max_d = max(1.0, float(analysis_res.get('max_depth_cm', depth_cm.max())))
        norm_depth = np.clip((depth_cm / max_d) * 255.0, 0, 255).astype(np.uint8)
        cm_id = resolve_colormap(colormap)
        heatmap = cv2.applyColorMap(norm_depth, cm_id)
        # Suppress non-water background to dark navy
        heatmap[pred_mask != 2] = [15, 23, 42]
        self._draw_interpretation_sign(heatmap, analysis_res)
        return heatmap

    def _draw_interpretation_sign(self, heatmap: np.ndarray, analysis_res: dict):
        """
        Draws an informative HUD sign / estimator box directly onto the depth heatmap image
        to help users interpret flood depth values and vehicle passability.
        """
        h, w = heatmap.shape[:2]
        max_d = float(analysis_res.get('max_depth_cm', 0.0))
        status = analysis_res.get('status', 'Passable')

        # Vehicle passability evaluations
        if max_d < 15.0:
            sedan_status = "PASSABLE"
            sedan_col = (50, 205, 50)      # Green
            suv_status = "PASSABLE"
            suv_col = (50, 205, 50)
            truck_status = "PASSABLE"
            truck_col = (50, 205, 50)
        elif max_d < 30.0:
            sedan_status = "AVOID (Hydrolock Hazard)"
            sedan_col = (0, 0, 235)        # Red
            suv_status = "CAUTION (Clearance Req)"
            suv_col = (0, 165, 255)        # Amber
            truck_status = "PASSABLE"
            truck_col = (50, 205, 50)
        else:
            sedan_status = "IMPASSABLE"
            sedan_col = (0, 0, 235)
            suv_status = "DANGER (Exceeds Clearance)"
            suv_col = (0, 0, 235)
            truck_status = "CAUTION (Emergency Only)"
            truck_col = (0, 165, 255)

        # Draw Sign Card on Top-Right of Heatmap
        sign_w = min(360, int(w * 0.48))
        sign_h = max(80, int(h * 0.22))
        sx = w - sign_w - 12
        sy = 10

        if sx > 150 and sign_h > 50:
            # Semi-transparent dark glassmorphic box
            box_region = heatmap[sy:sy + sign_h, sx:sx + sign_w].copy()
            black_box = np.zeros_like(box_region)
            heatmap[sy:sy + sign_h, sx:sx + sign_w] = cv2.addWeighted(box_region, 0.2, black_box, 0.8, 0)
            cv2.rectangle(heatmap, (sx, sy), (sx + sign_w, sy + sign_h), (59, 130, 246), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = max(0.35, h / 1200.0)

            # Header
            cv2.putText(heatmap, "ESTIMATOR GUIDE [Bathymetric Water Depth]", (sx + 8, sy + int(sign_h * 0.20)),
                        font, fs * 0.95, (200, 230, 255), 1, cv2.LINE_AA)

            # Est Peak Depth line
            status_col = analysis_res.get('color_bgr', (50, 205, 50))
            depth_line = f"Est. Peak: {max_d} cm  |  {status.upper()}"
            cv2.putText(heatmap, depth_line, (sx + 8, sy + int(sign_h * 0.42)),
                        font, fs * 1.05, status_col, 1, cv2.LINE_AA)

            # Vehicle Clearance Lines
            sedan_text = f"Sedan (<15cm): {sedan_status}"
            cv2.putText(heatmap, sedan_text, (sx + 8, sy + int(sign_h * 0.64)),
                        font, fs * 0.88, sedan_col, 1, cv2.LINE_AA)

            suv_text = f"SUV (<30cm): {suv_status}"
            cv2.putText(heatmap, suv_text, (sx + 8, sy + int(sign_h * 0.86)),
                        font, fs * 0.88, suv_col, 1, cv2.LINE_AA)

    def generate_visual_overlay(self, original_bgr, pred_mask, analysis_res, alpha=0.50):
        """
        Generates a visually striking HUD overlay with depth heatmap over floodwater,
        road geometry outline, and passability telemetry.
        """
        if original_bgr is None:
            # Return a blank placeholder if image is unavailable
            h, w = pred_mask.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        h, w, c = original_bgr.shape
        overlay = original_bgr.copy()

        # 1. Overlay Dry Road in Emerald Green (vectorized in-place blend)
        road_mask = (pred_mask == 1)
        if np.any(road_mask):
            overlay[road_mask] = (
                original_bgr[road_mask].astype(np.uint16) // 2 + np.array([25, 102, 25], dtype=np.uint16)
            ).astype(np.uint8)

        # 2. Overlay Floodwater with Turbo Depth Heatmap (vectorized in-place blend)
        depth_cm = analysis_res.get('_depth_cm_map')
        water_mask = (pred_mask == 2)
        if np.any(water_mask):
            if depth_cm is not None and analysis_res.get('max_depth_cm', 0) > 0:
                depth_norm = np.clip((depth_cm / max(1.0, analysis_res['max_depth_cm'])) * 255.0, 0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                overlay[water_mask] = (
                    (original_bgr[water_mask].astype(np.uint16) + heatmap[water_mask].astype(np.uint16)) >> 1
                ).astype(np.uint8)
            else:
                overlay[water_mask] = (
                    original_bgr[water_mask].astype(np.uint16) // 2 + np.array([0, 70, 127], dtype=np.uint16)
                ).astype(np.uint8)

        # 3. Draw HUD Panel at top
        panel_h = max(90, int(h * 0.17))
        hud_bg = overlay[0:panel_h, 0:w].copy()
        black_box = np.zeros_like(hud_bg)
        overlay[0:panel_h, 0:w] = cv2.addWeighted(hud_bg, 0.2, black_box, 0.8, 0)

        # HUD Text details
        depth_src = analysis_res.get('depth_source', 'morphological')
        if depth_src == 'sturm_flooddepth':
            lvl = analysis_res.get('sturm_level', 0)
            rng = analysis_res.get('sturm_depth_range_str', '')
            depth_label = f"STURM L{lvl} ({rng})"
            badge_color = analysis_res.get('color_bgr', (0, 200, 150))
        else:
            depth_label = "Morphological"
            badge_color = (120, 120, 120)

        status_text  = f"STATUS: {analysis_res['status'].upper()}"
        metrics_text = (
            f"Flooded Ratio: {analysis_res['flooded_road_ratio']}%  |  "
            f"Peak Depth: {analysis_res.get('max_depth_cm', 0.0)} cm  |  "
            f"Mean: {analysis_res.get('mean_depth_cm', 0.0)} cm  [{depth_label}]"
        )
        rec_text = f"Advisory: {analysis_res['recommendation']}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, h / 720.0)
        thickness = 2

        # Draw Status with color code
        cv2.putText(overlay, status_text,  (20, int(panel_h * 0.28)), font, font_scale * 1.05, analysis_res['color_bgr'], thickness)
        cv2.putText(overlay, metrics_text, (20, int(panel_h * 0.58)), font, font_scale * 0.78, (255, 255, 255), 1)
        cv2.putText(overlay, rec_text,     (20, int(panel_h * 0.84)), font, font_scale * 0.68, (200, 220, 240), 1)

        # Depth source badge (top-right corner of HUD)
        badge_text  = f"Depth: {depth_label}"
        (bw, bh), _ = cv2.getTextSize(badge_text, font, font_scale * 0.60, 1)
        bx = w - bw - 20
        by = int(panel_h * 0.35)
        cv2.rectangle(overlay, (bx - 6, by - bh - 4), (bx + bw + 6, by + 4), badge_color, -1)
        cv2.putText(overlay, badge_text, (bx, by), font, font_scale * 0.60, (0, 0, 0), 1)

        # 4. Draw Legend at bottom right
        legend_w, legend_h = 260, 55
        cv2.rectangle(overlay, (w - legend_w - 10, h - legend_h - 10), (w - 10, h - 10), (25, 25, 25), -1)
        cv2.rectangle(overlay, (w - legend_w + 5, h - legend_h + 5),  (w - legend_w + 25, h - legend_h + 20), (50, 205, 50), -1)
        cv2.putText(overlay, "Dry Road", (w - legend_w + 32, h - legend_h + 18), font, 0.42, (255, 255, 255), 1)
        cv2.rectangle(overlay, (w - legend_w + 5, h - legend_h + 30), (w - legend_w + 25, h - legend_h + 45), (0, 140, 255), -1)
        cv2.putText(overlay, f"Water Depth (0->{int(self.max_allowable_depth_cm)}+cm)", (w - legend_w + 32, h - legend_h + 43), font, 0.42, (255, 255, 255), 1)

        return overlay

