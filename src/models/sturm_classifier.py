"""
STURM-FloodDepth Flood Level Classifier
=======================================
Implements the MobileNetV3-Small deep learning model for classifying flood depth
into 5 discrete levels (Level 0 to Level 4) based on the STURM-FloodDepth dataset:
"STURM-FloodDepth Flooded Cars" (Zenodo DOI: 10.5281/zenodo.14833532).

Levels & Calibrated Physical Depths:
  - Level 0: Dry / No Flooding (0 cm)
  - Level 1: Ankle / Sub-axle depth (< 15 cm) — below half vehicle tire
  - Level 2: Wheel hub / Half-to-full tire depth (15 - 30 cm)
  - Level 3: Chassis / Exhaust / Door-sill / Window-base depth (30 - 60 cm)
  - Level 4: Deep inundation / Submerged (> 60 cm) — exceeds window line
"""

import torch
import torch.nn as nn
from torchvision import models

# Class definitions and physical telemetry mapping
STURM_CLASSES = ['Level0', 'Level1', 'Level2', 'Level3', 'Level4']

STURM_LEVEL_METADATA = {
    0: {
        'level': 0,
        'label': 'Level 0',
        'title': 'Dry / Wheels on Ground',
        'depth_min_cm': 0.0,
        'depth_max_cm': 10.0,
        'est_depth_cm': 5.0,
        'landmark': 'Dry / Wheels fully on ground',
        'passability': 'Passable',
        'hazard': 'Low Risk (Dry / Minimal Water)',
        'color_bgr': (46, 204, 113),    # Green
        'color_hex': '#10B981',
        'badge_class': 'passable',
        'recommendation': 'Road is dry and clear. Safe for all vehicles. Normal conditions.',
    },
    1: {
        'level': 1,
        'label': 'Level 1',
        'title': 'Lower Tire / Tread',
        'depth_min_cm': 10.0,
        'depth_max_cm': 25.0,
        'est_depth_cm': 18.0,
        'landmark': 'Water touching lower tire / tread',
        'passability': 'Passable',
        'hazard': 'Low Risk (Safe for most vehicles)',
        'color_bgr': (46, 204, 113),    # Green
        'color_hex': '#10B981',
        'badge_class': 'passable',
        'recommendation': 'Minor water touching lower tire (10-25 cm). Safe for most vehicles.',
    },
    2: {
        'level': 2,
        'label': 'Level 2',
        'title': 'Wheel Center / Door Sill',
        'depth_min_cm': 25.0,
        'depth_max_cm': 40.0,
        'est_depth_cm': 32.0,
        'landmark': 'Water at wheel center / door sill',
        'passability': 'Partially Flooded',
        'hazard': 'Caution: SUVs / Trucks only',
        'color_bgr': (0, 165, 255),    # Amber
        'color_hex': '#F59E0B',
        'badge_class': 'partially-flooded',
        'recommendation': 'CAUTION: Water at wheel center (25-40 cm). SUVs / Emergency trucks only. Sedans avoid.',
    },
    3: {
        'level': 3,
        'label': 'Level 3',
        'title': 'Top of Tire / Wheel Arch',
        'depth_min_cm': 40.0,
        'depth_max_cm': 60.0,
        'est_depth_cm': 50.0,
        'landmark': 'Water at top of tire / wheel arch',
        'passability': 'Severely Flooded',
        'hazard': 'Unsafe: Engine flood hazard',
        'color_bgr': (0, 0, 235),      # Red
        'color_hex': '#EF4444',
        'badge_class': 'severely-affected',
        'recommendation': 'UNSAFE: Water at top of tire (40-60 cm). Danger of engine stalling / air intake flood. Reroute!',
    },
    4: {
        'level': 4,
        'label': 'Level 4',
        'title': 'Submerged Past Hood / Windows',
        'depth_min_cm': 60.0,
        'depth_max_cm': 120.0,
        'est_depth_cm': 85.0,
        'landmark': 'Submerged past hood / windows',
        'passability': 'Severely Flooded',
        'hazard': 'Extreme Hazard: Submerged',
        'color_bgr': (0, 0, 180),      # Deep Crimson
        'color_hex': '#DC2626',
        'badge_class': 'severely-affected',
        'recommendation': 'DISASTER ZONE: Deep inundation (>60 cm). Vehicles submerged. Completely impassable!',
    },
}


class STURMFloodLevelClassifier(nn.Module):
    """
    MobileNetV3-Small based classifier for the STURM-FloodDepth 5-class flood levels.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        
        # Load torchvision MobileNetV3-Small backbone
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Replace classifier head to match STURM 5 classes
        in_features = base_model.classifier[3].in_features  # 1024
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        self.name = f"STURM_MobileNetV3_Small_{num_classes}cls"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

