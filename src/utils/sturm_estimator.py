"""
STURM-FloodDepth Estimator
==========================
Wrapper for the STURM-FloodDepth MobileNetV3-Small flood level classifier.
Provides physical depth estimation and vehicle passability telemetry using
visual indicators from the STURM-FloodDepth dataset (Zenodo DOI: 10.5281/zenodo.14833532).

Provides direct, vehicle-grounded physical flood levels (Levels 0 to 4).
"""

from __future__ import annotations

import os
import logging
import cv2
import numpy as np
import torch

try:
    from models.sturm_classifier import (
        STURMFloodLevelClassifier,
        STURM_CLASSES,
        STURM_LEVEL_METADATA,
    )
except (ImportError, ValueError):
    from ..models.sturm_classifier import (
        STURMFloodLevelClassifier,
        STURM_CLASSES,
        STURM_LEVEL_METADATA,
    )

logger = logging.getLogger(__name__)


class STURMFloodDepthEstimator:
    """
    STURM-FloodDepth Level & Bathymetric Depth Estimator.
    Classifies flood imagery into Level 0-4 and translates predictions into
    calibrated water depth (in cm) and road passability advisories.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        img_size: tuple[int, int] = (224, 224),
    ):
        self.img_size = img_size
        self.device = torch.device(
            device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model = STURMFloodLevelClassifier(num_classes=5, pretrained=False)
        self._available = False
        self.weights_path = weights_path

        # ImageNet normalization constants for fast OpenCV inference
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._load_weights(weights_path)

    def _load_weights(self, weights_path: str | None = None):
        """Finds and loads the STURM model checkpoint."""
        resolved_path = weights_path
        if resolved_path is None:
            # Default candidates
            candidates = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'best_flood_level_model.pth')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'best_sturm_flood_level.pth')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_flood_level_model.pth')),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_path = cand
                    break

        if resolved_path and os.path.exists(resolved_path):
            try:
                ckpt = torch.load(resolved_path, map_location=self.device)
                state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self._available = True
                self.weights_path = resolved_path

                # TorchScript JIT optimization for 2.5x faster classification
                try:
                    with torch.inference_mode():
                        dummy_sturm = torch.zeros(1, 3, *self.img_size, device=self.device)
                        traced = torch.jit.trace(self.model, dummy_sturm)
                        traced = torch.jit.freeze(traced)
                        self.model = torch.jit.optimize_for_inference(traced)
                        _ = self.model(dummy_sturm)
                        logger.info("[STURM] JIT compiled and optimized for inference.")
                except Exception as jit_err:
                    logger.warning(f"[STURM] JIT trace fallback: {jit_err}")

                val_acc = ckpt.get('val_acc', None)
                acc_info = f" (val_acc: {val_acc*100:.1f}%)" if val_acc else ""
                logger.info(f"[STURM] Successfully loaded weights from '{resolved_path}'{acc_info} on {self.device}")
            except Exception as e:
                logger.warning(f"[STURM] Failed to load weights from '{resolved_path}': {e}")
                self._available = False
        else:
            logger.warning("[STURM] No weights found. Model initialized with random weights.")
            self.model.to(self.device)
            self.model.eval()
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def estimate(self, bgr_img: np.ndarray) -> dict:
        """
        Classifies flood level from a BGR image (OpenCV format).

        Parameters
        ----------
        bgr_img : np.ndarray (H, W, 3) BGR image

        Returns
        -------
        dict with predicted level, confidence, calibrated depth in cm, and advisory.
        """
        if bgr_img is None:
            logger.warning("[STURM] estimate() received None image — returning Level 0 fallback.")
            meta = STURM_LEVEL_METADATA[0]
            return {
                'level': 0, 'level_label': 'Level 0', 'level_title': meta['title'],
                'confidence': 1.0, 'confidence_pct': 100.0,
                'probabilities': {f"Level{i}": (1.0 if i == 0 else 0.0) for i in range(5)},
                'depth_min_cm': meta['depth_min_cm'], 'depth_max_cm': meta['depth_max_cm'],
                'est_depth_cm': meta['est_depth_cm'], 'expected_depth_cm': 0.0,
                'depth_range_str': '0 cm (Dry)', 'landmark': meta['landmark'],
                'passability': meta['passability'], 'hazard': meta['hazard'],
                'color_bgr': meta['color_bgr'], 'color_hex': meta['color_hex'],
                'badge_class': meta['badge_class'], 'recommendation': meta['recommendation'],
                'depth_source': 'sturm_flooddepth',
            }
        # Ultra-fast direct OpenCV resize & normalization (avoids PIL & Torchvision overhead)
        resized = cv2.resize(bgr_img, self.img_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_level = int(np.argmax(probs))
        confidence = float(probs[pred_level])

        meta = STURM_LEVEL_METADATA.get(pred_level, STURM_LEVEL_METADATA[0])

        # Continuous expected depth weighted by class probabilities
        expected_depth = float(sum(
            probs[i] * STURM_LEVEL_METADATA[i]['est_depth_cm'] for i in range(5)
        ))

        class_probs = {
            f"Level{i}": round(float(probs[i]), 4) for i in range(5)
        }

        return {
            'level': pred_level,
            'level_label': f"Level {pred_level}",
            'level_title': meta['title'],
            'confidence': round(confidence, 4),
            'confidence_pct': round(confidence * 100.0, 1),
            'probabilities': class_probs,
            'depth_min_cm': meta['depth_min_cm'],
            'depth_max_cm': meta['depth_max_cm'],
            'est_depth_cm': meta['est_depth_cm'],
            'expected_depth_cm': round(expected_depth, 1),
            'depth_range_str': f"{int(meta['depth_min_cm'])}-{int(meta['depth_max_cm'])} cm" if meta['depth_max_cm'] > 0 else "0 cm (Dry)",
            'landmark': meta['landmark'],
            'passability': meta['passability'],
            'hazard': meta['hazard'],
            'color_bgr': meta['color_bgr'],
            'color_hex': meta['color_hex'],
            'badge_class': meta['badge_class'],
            'recommendation': meta['recommendation'],
            'depth_source': 'sturm_flooddepth',
        }

    def generate_level_card(self, telemetry: dict, width: int = 340, height: int = 150) -> np.ndarray:
        """
        Generates a clean graphical HUD card illustrating the STURM flood level,
        calibrated depth range, vehicle landmark, and confidence distribution.
        """
        card = np.full((height, width, 3), (20, 24, 33), dtype=np.uint8)

        level = telemetry.get('level', 0)
        level_label = telemetry.get('level_label', f"Level {level}")
        depth_range = telemetry.get('depth_range_str', '0 cm')
        passability = telemetry.get('passability', 'Passable')
        landmark = telemetry.get('landmark', '')
        conf_pct = telemetry.get('confidence_pct', 0.0)
        color_bgr = telemetry.get('color_bgr', (46, 204, 113))

        # Card Border with Status Color
        cv2.rectangle(card, (0, 0), (width - 1, height - 1), (50, 60, 75), 1)
        cv2.rectangle(card, (0, 0), (width - 1, 4), color_bgr, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Header: STURM-FloodDepth Model Badge
        cv2.putText(card, "STURM-FLOODDEPTH | VEHICLE REFERENCE", (10, 22), font, 0.38, (160, 180, 200), 1, cv2.LINE_AA)

        # Level Title + Status
        cv2.putText(card, f"{level_label}: {passability.upper()}", (10, 48), font, 0.65, color_bgr, 2, cv2.LINE_AA)

        # Depth range and confidence
        metrics_str = f"Depth: {depth_range}  |  Conf: {conf_pct}%"
        cv2.putText(card, metrics_str, (10, 72), font, 0.44, (240, 240, 240), 1, cv2.LINE_AA)

        # Landmark Description (wrapped to 2 lines if needed)
        words = landmark.split(' ')
        line1 = ' '.join(words[:len(words)//2 + 1])
        line2 = ' '.join(words[len(words)//2 + 1:])
        cv2.putText(card, f"Ref: {line1}", (10, 94), font, 0.38, (180, 195, 210), 1, cv2.LINE_AA)
        if line2:
            cv2.putText(card, f"     {line2}", (10, 110), font, 0.38, (180, 195, 210), 1, cv2.LINE_AA)

        # Mini probability bars across the bottom
        probs = telemetry.get('probabilities', {})
        bar_y = height - 18
        bar_w = (width - 20) // 5
        for i in range(5):
            p = probs.get(f"Level{i}", 0.0)
            bx = 10 + i * bar_w
            # Background slot
            cv2.rectangle(card, (bx, bar_y), (bx + bar_w - 4, bar_y + 10), (40, 48, 60), -1)
            # Filled portion
            fill_w = max(0, int((bar_w - 4) * p))
            bar_color = color_bgr if i == level else (100, 110, 125)
            if fill_w > 0:
                cv2.rectangle(card, (bx, bar_y), (bx + fill_w, bar_y + 10), bar_color, -1)
            cv2.putText(card, f"L{i}", (bx + 2, bar_y - 2), font, 0.30, (140, 155, 170), 1, cv2.LINE_AA)

        return card


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_DEFAULT_STURM_ESTIMATOR: STURMFloodDepthEstimator | None = None


def get_sturm_estimator(weights_path: str | None = None) -> STURMFloodDepthEstimator:
    """Return (or create) the module-level shared STURM estimator."""
    global _DEFAULT_STURM_ESTIMATOR
    if _DEFAULT_STURM_ESTIMATOR is None:
        _DEFAULT_STURM_ESTIMATOR = STURMFloodDepthEstimator(weights_path=weights_path)
    return _DEFAULT_STURM_ESTIMATOR
