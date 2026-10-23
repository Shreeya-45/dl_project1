import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_flood_sample(img_id, out_img_dir, out_mask_dir, width=512, height=384):
    """
    Generates a realistic synthetic flood scene with corresponding 3-class ground truth mask:
    0: Background (Buildings, Trees, Sky, Sidewalk)
    1: Dry Road
    2: Flooded Road / Waterlogging
    """
    # Initialize Canvas
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8) # 0 = Background

    # 1. Background Scene (Sky, Vegetation, Buildings)
    sky_h = int(height * random.uniform(0.25, 0.35))
    # Sky Gradient
    sky_color1 = np.array([200, 180, 160]) # Overcast sky BGR
    sky_color2 = np.array([160, 140, 120])
    for y in range(sky_h):
        r = y / sky_h
        img[y, :] = (sky_color1 * (1 - r) + sky_color2 * r).astype(np.uint8)

    # Buildings / Background Trees
    bg_y = sky_h
    road_start_y = int(height * random.uniform(0.40, 0.50))
    img[bg_y:road_start_y, :] = [60, 80, 70] # Dark foliage/structures

    # Add background texture/noise
    bg_noise = np.random.normal(0, 10, (road_start_y - bg_y, width, 3))
    img[bg_y:road_start_y, :] = np.clip(img[bg_y:road_start_y, :] + bg_noise, 0, 255).astype(np.uint8)

    # 2. Road Surface Geometry (Perspective Trapezoid / Roadway)
    road_top_left = (int(width * random.uniform(0.3, 0.4)), road_start_y)
    road_top_right = (int(width * random.uniform(0.6, 0.7)), road_start_y)
    road_bottom_left = (int(width * random.uniform(0.0, 0.1)), height)
    road_bottom_right = (int(width * random.uniform(0.9, 1.0)), height)

    road_pts = np.array([road_top_left, road_top_right, road_bottom_right, road_bottom_left], np.int32)
    
    # Asphalt Texture BGR
    asphalt_color = np.array([80, 85, 90])
    cv2.fillPoly(img, [road_pts], asphalt_color.tolist())
    cv2.fillPoly(mask, [road_pts], 1) # Class 1: Dry Road

    # Add asphalt noise
    road_mask_bool = (mask == 1)
    asphalt_noise = np.random.normal(0, 12, (height, width, 3))
    img[road_mask_bool] = np.clip(img[road_mask_bool] + asphalt_noise[road_mask_bool], 0, 255).astype(np.uint8)

    # Lane markings (Yellow/White center line)
    lane_pts = np.array([
        (int((road_top_left[0] + road_top_right[0])/2), road_start_y),
        (int((road_bottom_left[0] + road_bottom_right[0])/2), height)
    ], np.int32)
    cv2.polylines(img, [lane_pts], isClosed=False, color=(50, 200, 240), thickness=3)

    # 3. Flood Waterlogging Patterns (Varied levels: low, medium, heavy)
    flood_severity = random.choice(['light', 'moderate', 'severe', 'dry'])

    if flood_severity != 'dry':
        # Generate organic water region using random ellipses and contours
        water_mask = np.zeros((height, width), dtype=np.uint8)
        num_patches = random.randint(1, 4)
        
        if flood_severity == 'severe':
            # Submerge most of bottom road
            water_pts = np.array([
                (int(width * 0.1), int(height * 0.55)),
                (int(width * 0.9), int(height * 0.52)),
                road_bottom_right,
                road_bottom_left
            ], np.int32)
            cv2.fillPoly(water_mask, [water_pts], 255)
        else:
            for _ in range(num_patches):
                cx = random.randint(int(width * 0.2), int(width * 0.8))
                cy = random.randint(int(height * 0.55), int(height * 0.9))
                axes = (random.randint(60, 180), random.randint(30, 90))
                angle = random.randint(-20, 20)
                cv2.ellipse(water_mask, (cx, cy), axes, angle, 0, 360, 255, -1)

        # Smooth water contours
        water_mask = cv2.GaussianBlur(water_mask, (15, 15), 0)
        _, water_mask_thresh = cv2.threshold(water_mask, 100, 255, cv2.THRESH_BINARY)
        
        # Water is only where road or road margin exists
        flood_on_road = (water_mask_thresh > 0) & (mask == 1)
        mask[flood_on_road] = 2 # Class 2: Water / Flood

        # Render Water Texture & Reflections
        water_color = np.array([120, 110, 80]) # Muddy brownish/cyan water
        water_indices = (mask == 2)
        
        # Water ripples / reflections
        water_layer = img.copy()
        water_layer[water_indices] = water_color
        # Add specular reflections
        reflection_noise = np.random.normal(0, 20, (height, width, 3))
        water_layer[water_indices] = np.clip(water_layer[water_indices] + reflection_noise[water_indices], 0, 255).astype(np.uint8)

        # Blend water onto image
        img[water_indices] = cv2.addWeighted(img[water_indices], 0.35, water_layer[water_indices], 0.65, 0)

    # 4. Save Synthetic Files
    img_filename = f"sample_{img_id:04d}.jpg"
    mask_filename = f"sample_{img_id:04d}.png"

    img_path = os.path.join(out_img_dir, img_filename)
    mask_path = os.path.join(out_mask_dir, mask_filename)

    cv2.imwrite(img_path, img)
    cv2.imwrite(mask_path, mask)


def main():
    print("Generating Synthetic Multi-Scene Flood & Road Dataset for Capstone Project...")
    raw_dir = os.path.join("data", "raw")
    img_dir = os.path.join(raw_dir, "images")
    mask_dir = os.path.join(raw_dir, "masks")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    num_samples = 200
    for i in range(1, num_samples + 1):
        create_synthetic_flood_sample(i, img_dir, mask_dir)

    print(f"Successfully generated {num_samples} sample images & ground truth masks in '{raw_dir}'!")

if __name__ == "__main__":
    main()
