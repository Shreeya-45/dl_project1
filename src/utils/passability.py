import cv2
import numpy as np

class PassabilityAnalyzer:
    """
    Road Passability and Morphological Flood Depth Analysis Engine.
    Estimates road passability using Flooded-Road Pixel Ratio, Connected Road Geometry,
    and Euclidean Distance Transform (EDT) bathymetric flood depth approximation.
    Classifies scenes into: Passable, Partially Flooded, Severely Affected.
    """
    def __init__(self, passable_thresh=0.15, severe_thresh=0.50, max_allowable_depth_cm=50.0):
        self.passable_thresh = passable_thresh
        self.severe_thresh = severe_thresh
        self.max_allowable_depth_cm = max_allowable_depth_cm

    def compute_depth_map(self, pred_mask, img_shape):
        """
        Calculates morphological water depth (in cm) from the 2D water mask (pred_mask == 2).
        Returns:
            depth_cm: 2D float array with depth in cm for each pixel.
            max_depth: peak depth in cm.
            mean_depth: average depth of flooded pixels in cm.
        """
        h, w = img_shape[:2]
        water_binary = (pred_mask == 2).astype(np.uint8)

        if np.sum(water_binary) == 0:
            return np.zeros((h, w), dtype=np.float32), 0.0, 0.0

        # 1. Compute Euclidean Distance Transform (L2 metric)
        dist_transform = cv2.distanceTransform(water_binary, cv2.DIST_L2, 5)
        d_max = dist_transform.max()

        if d_max == 0:
            return np.zeros((h, w), dtype=np.float32), 0.0, 0.0

        # 2. Scale peak depth dynamically based on pool width relative to scene width
        pool_scale = min(1.0, (d_max * 2.0) / (w * 0.45))
        peak_depth_cm = max(10.0, pool_scale * self.max_allowable_depth_cm)

        # 3. Concave bathymetric mapping: Depth = Peak * (dist / d_max)^0.75
        normalized_dist = (dist_transform / d_max) ** 0.75
        depth_cm = normalized_dist * peak_depth_cm
        depth_cm[water_binary == 0] = 0.0

        water_pixels = depth_cm[water_binary == 1]
        mean_depth = float(np.mean(water_pixels)) if len(water_pixels) > 0 else 0.0
        max_depth = float(depth_cm.max())

        return depth_cm, round(max_depth, 1), round(mean_depth, 1)

    def analyze(self, pred_mask):
        """
        pred_mask: 2D numpy array with class indices (0: Background, 1: Dry Road, 2: Water/Flood)
        Returns dictionary containing passability status, ratios, depth telemetry, and connected geometry.
        """
        h, w = pred_mask.shape[:2]
        road_pixels = int(np.sum(pred_mask == 1))
        flood_pixels = int(np.sum(pred_mask == 2))

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

        # Depth-Aware Passability Classification Rules:
        # Severe / Impassable:
        #   - Flooded road ratio >= 50% (severe inundation), OR
        #   - High water depth (>= 30.0 cm) with non-negligible pool coverage (ratio >= 20%)
        # Partially Flooded:
        #   - Flooded road ratio >= 15%, OR
        #   - Moderate water depth (>= 15.0 cm)
        # Passable:
        #   - Low water (ratio < 15% and depth < 15.0 cm)
        if flooded_ratio >= self.severe_thresh or (max_depth >= 30.0 and flooded_ratio >= 0.20):
            status = "Severely Affected"
            color_bgr = (0, 0, 235)     # Deep Red
            color_hex = "#EF4444"
            recommendation = f"UNSAFE: High water depth ({max_depth} cm) or severe inundation. Engine flood hazard. Reroute traffic!"
            badge_class = "severely-affected"
        elif flooded_ratio >= self.passable_thresh or max_depth >= 15.0:
            status = "Partially Flooded"
            color_bgr = (0, 165, 255)   # Orange / Amber
            color_hex = "#F59E0B"
            recommendation = f"CAUTION: Water depth {max_depth} cm. Clearance required (SUVs / Emergency Trucks only). Sedans avoid."
            badge_class = "partially-flooded"
        else:
            status = "Passable"
            color_bgr = (46, 204, 113)  # Green
            color_hex = "#10B981"
            recommendation = f"Safe to drive with normal caution. Low water level ({max_depth} cm)."
            badge_class = "passable"

        # Check if road geometry is fragmented
        road_fragmented = num_road_segments > 2 and flooded_ratio > 0.10

        return {
            'status': status,
            'flooded_road_ratio': round(flooded_ratio * 100, 2),
            'max_depth_cm': max_depth,
            'mean_depth_cm': mean_depth,
            'flooded_pixels': int(flood_pixels),
            'dry_road_pixels': int(dry_road_pixels),
            'total_road_pixels': int(total_road_surface),
            'road_segments_count': num_road_segments,
            'is_fragmented': road_fragmented,
            'color_bgr': color_bgr,
            'color_hex': color_hex,
            'recommendation': recommendation,
            'badge_class': badge_class,
            '_depth_cm_map': depth_cm
        }

    def generate_visual_overlay(self, original_bgr, pred_mask, analysis_res, alpha=0.50):
        """
        Generates a visually striking HUD overlay with depth heatmap over floodwater,
        road geometry outline, and passability telemetry.
        """
        h, w, c = original_bgr.shape
        overlay = original_bgr.copy()

        # 1. Overlay Dry Road in Emerald Green
        road_mask = (pred_mask == 1)
        if np.any(road_mask):
            overlay[road_mask] = cv2.addWeighted(
                original_bgr[road_mask], 1.0 - alpha,
                np.full_like(original_bgr[road_mask], (50, 205, 50)), alpha, 0
            )

        # 2. Overlay Floodwater with Turbo Depth Heatmap
        depth_cm = analysis_res.get('_depth_cm_map')
        water_mask = (pred_mask == 2)
        if np.any(water_mask):
            if depth_cm is not None and analysis_res.get('max_depth_cm', 0) > 0:
                # Normalize water depth to [0, 255]
                depth_norm = np.clip((depth_cm / max(1.0, analysis_res['max_depth_cm'])) * 255.0, 0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                overlay[water_mask] = cv2.addWeighted(
                    original_bgr[water_mask], 1.0 - alpha,
                    heatmap[water_mask], alpha, 0
                )
            else:
                overlay[water_mask] = cv2.addWeighted(
                    original_bgr[water_mask], 1.0 - alpha,
                    np.full_like(original_bgr[water_mask], (0, 140, 255)), alpha, 0
                )

        # 3. Draw HUD Panel at top
        panel_h = max(75, int(h * 0.15))
        hud_bg = overlay[0:panel_h, 0:w].copy()
        black_box = np.zeros_like(hud_bg)
        overlay[0:panel_h, 0:w] = cv2.addWeighted(hud_bg, 0.2, black_box, 0.8, 0)

        # HUD Text details
        status_text = f"STATUS: {analysis_res['status'].upper()}"
        metrics_text = f"Flooded Ratio: {analysis_res['flooded_road_ratio']}%  |  Est. Peak Depth: {analysis_res.get('max_depth_cm', 0.0)} cm  |  Mean: {analysis_res.get('mean_depth_cm', 0.0)} cm"
        rec_text = f"Advisory: {analysis_res['recommendation']}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, h / 720.0)
        thickness = 2

        # Draw Status with color code
        cv2.putText(overlay, status_text, (20, int(panel_h * 0.32)), font, font_scale * 1.05, analysis_res['color_bgr'], thickness)
        cv2.putText(overlay, metrics_text, (20, int(panel_h * 0.64)), font, font_scale * 0.82, (255, 255, 255), 1)
        cv2.putText(overlay, rec_text, (20, int(panel_h * 0.90)), font, font_scale * 0.70, (200, 220, 240), 1)

        # 4. Draw Legend at bottom right
        legend_w, legend_h = 240, 55
        cv2.rectangle(overlay, (w - legend_w - 10, h - legend_h - 10), (w - 10, h - 10), (25, 25, 25), -1)
        cv2.rectangle(overlay, (w - legend_w + 5, h - legend_h + 5), (w - legend_w + 25, h - legend_h + 20), (50, 205, 50), -1)
        cv2.putText(overlay, "Dry Road", (w - legend_w + 32, h - legend_h + 18), font, 0.42, (255, 255, 255), 1)
        cv2.rectangle(overlay, (w - legend_w + 5, h - legend_h + 30), (w - legend_w + 25, h - legend_h + 45), (0, 140, 255), -1)
        cv2.putText(overlay, "Water Depth (0->30+cm)", (w - legend_w + 32, h - legend_h + 43), font, 0.42, (255, 255, 255), 1)

        return overlay

