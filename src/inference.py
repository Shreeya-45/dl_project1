import os
import argparse
import time
import torch
import cv2
import numpy as np
from PIL import Image

from models import UNetBaseline, MobileNetUNet
from utils import PassabilityAnalyzer

class FloodSegmentationPredictor:
    """
    Inference and Telemetry Pipeline for Flood/Road Segmentation & Passability Estimation.
    """
    def __init__(self, model_type='mobilenet_unet', weights_path=None, img_size=(256, 256)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.model_type = model_type.lower()

        if self.model_type == 'unet':
            self.model = UNetBaseline(in_channels=3, num_classes=3)
        else:
            self.model = MobileNetUNet(in_channels=3, num_classes=3)

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained weights from '{weights_path}'")
        else:
            print(f"Running model '{self.model.name}' with initialized weights.")

        self.model.to(self.device)
        self.model.eval()
        self.analyzer = PassabilityAnalyzer()

    def predict_image(self, bgr_img):
        """
        Takes input BGR image (OpenCV format) and runs end-to-end segmentation & passability analysis.
        Returns: pred_mask (H, W), overlay_bgr (H, W, 3), analysis_summary (dict)
        """
        h_orig, w_orig, _ = bgr_img.shape

        # Preprocessing
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb_img, self.img_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.tensor(resized_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Model Inference & Latency measurement
        start_t = time.time()
        with torch.no_grad():
            logits = self.model(img_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        latency_ms = (time.time() - start_t) * 1000.0

        # Resize prediction mask back to original resolution
        pred_mask_orig = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Passability Analysis
        passability_info = self.analyzer.analyze(pred_mask_orig)
        passability_info['latency_ms'] = round(latency_ms, 2)
        passability_info['fps'] = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0

        # Generate HUD Visual Overlay
        overlay_bgr = self.analyzer.generate_visual_overlay(bgr_img, pred_mask_orig, passability_info)

        return pred_mask_orig, overlay_bgr, passability_info


def main():
    parser = argparse.ArgumentParser(description="Run Flood Segmentation & Road Passability Inference")
    parser.add_argument('--input', type=str, required=True, help='Path to input image file')
    parser.add_argument('--model', type=str, default='mobilenet_unet', choices=['unet', 'mobilenet_unet'])
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights checkpoint')
    parser.add_argument('--out', type=str, default='outputs/prediction.jpg', help='Output visualization path')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' does not exist.")
        return

    bgr_img = cv2.imread(args.input)
    predictor = FloodSegmentationPredictor(model_type=args.model, weights_path=args.weights)

    pred_mask, overlay, summary = predictor.predict_image(bgr_img)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cv2.imwrite(args.out, overlay)

    print("\n--- INFERENCE & PASSABILITY TELEmetry ---")
    print(f"Input Image Path : {args.input}")
    print(f"Status           : {summary['status']}")
    print(f"Flooded Ratio    : {summary['flooded_road_ratio']}%")
    print(f"Inference Latency: {summary['latency_ms']} ms ({summary['fps']} FPS)")
    print(f"Advisory         : {summary['recommendation']}")
    print(f"Saved HUD Overlay: {args.out}")

if __name__ == "__main__":
    main()
