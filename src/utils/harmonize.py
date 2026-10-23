"""
Label Harmonization Module
Harmonizes multi-dataset semantic labels into a unified class schema:
Class 0: Background (Non-flooded terrain, buildings, trees, sky)
Class 1: Road (Dry passable road surface)
Class 2: Flood / Water (Flooded road, standing waterlogging, submerged road)
"""

FLOODNET_LABEL_MAP = {
    0: 0, # Background / Sky / Vegetation
    1: 1, # Building -> Background
    2: 1, # Dry Road -> Road (1)
    3: 0, # Grass / Vegetation -> Background (0)
    4: 0, # Tree -> Background (0)
    5: 2, # Flooded Road -> Water (2)
    6: 2, # Water pool / Waterlogging -> Water (2)
    7: 0, # Vehicle -> Background (0)
    8: 0, # Pool -> Background (0)
    9: 0  # Asset -> Background (0)
}

CVFD_LABEL_MAP = {
    0: 0, # Background
    1: 1, # Road
    2: 2  # Water / Floodwater
}

FLOOD_DETECTION_V1_MAP = {
    0: 0, # Non-flooded / level0 -> Background (or dry road depending on mask)
    1: 2  # Flooded / level2 -> Water (2)
}

CLASS_NAMES = ["Background", "Road", "Water / Floodwater"]
CLASS_COLORS = [
    [40, 44, 52],    # Dark Grey for Background
    [16, 185, 129],  # Emerald Green for Passable Road
    [245, 158, 11]   # Amber / Bright Orange-Red for Floodwater
]

def harmonize_mask(raw_mask, dataset_source='floodnet'):
    """
    Harmonizes a raw mask numpy array into unified 0, 1, 2 classes.
    """
    source = dataset_source.lower()
    if 'floodnet' in source:
        mapping = FLOODNET_LABEL_MAP
    elif 'cvfd' in source:
        mapping = CVFD_LABEL_MAP
    else:
        mapping = FLOOD_DETECTION_V1_MAP

    harmonized = raw_mask.copy()
    for raw_val, target_val in mapping.items():
        harmonized[raw_mask == raw_val] = target_val
        
    return harmonized
