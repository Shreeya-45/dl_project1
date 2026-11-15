import os
import argparse
import time
import torch
import cv2
import numpy as np
from PIL import Image

from models import UNetBaseline, MobileNetUNet
from utils import PassabilityAnalyzer, get_sturm_estimator


class FloodSegmentationPredictor:
    """
    Inference and Telemetry Pipeline for Flood/Road Segmentation & Passability Estimation.

    Supports two depth backends:
      - STURM-FloodDepth (default/proposed, 5-class vehicle-grounded physical levels from Zenodo: 14833532)
      - Morphological Bathymetry (fallback; Euclidean Distance Transform)
    """
    def __init__(
        self,
        model_type: str = 'mobilenet_unet',
        weights_path: str | None = None,
        img_size: tuple = (256, 256),
        depth_backend: str = 'sturm',
        colormap: str = 'turbo',
        sturm_weights: str | None = None,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.model_type = model_type.lower()
        self.depth_backend = depth_backend.lower()
        self.colormap = colormap

        if self.model_type == 'unet':
            self.model = UNetBaseline(in_channels=3, num_classes=3)
        else:
            self.model = MobileNetUNet(in_channels=3, num_classes=3)

        if weights_path is None:
            default_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights', f"best_{self.model_type}.pth"))
            if os.path.exists(default_weights):
                weights_path = default_weights

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained weights from '{weights_path}'")
        else:
            print(f"Running model '{self.model.name}' with initialized weights.")

        self.model.to(self.device)
        self.model.eval()

        # Optimal CPU thread configuration (avoid thread contention on hybrid P/E core architectures)
        if self.device.type == 'cpu':
            torch.set_num_threads(min(6, os.cpu_count() or 4))

        # TorchScript JIT tracing and optimization (2-3x inference speedup on CPU and GPU)
        try:
            with torch.inference_mode():
                dummy_input = torch.zeros(1, 3, *self.img_size, device=self.device)
                traced = torch.jit.trace(self.model, dummy_input)
                traced = torch.jit.freeze(traced)
                self.model = torch.jit.optimize_for_inference(traced)
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                print(f"[{self.model_type}] JIT compiled and optimized for high-speed inference.")
        except Exception as e:
            print(f"[{self.model_type}] JIT trace fallback to eager mode: {e}")
            if self.device.type == 'cuda':
                with torch.no_grad():
                    _dummy = torch.zeros(1, 3, *self.img_size, device=self.device)
                    _ = self.model(_dummy)
                    torch.cuda.synchronize()

        self.analyzer = PassabilityAnalyzer()

        # STURM Depth Estimator backend initialisation
        self._sturm_estimator = None

        if self.depth_backend == 'sturm':
            print(f"[STURM] Initialising STURM-FloodDepth vehicle reference estimator …")
            self._sturm_estimator = get_sturm_estimator(weights_path=sturm_weights)

    def predict_image(
        self,
        bgr_img: np.ndarray,
        generate_heatmap: bool = True,
        sturm_telemetry: dict | None = None,
    ) -> tuple:
        """
        Takes input BGR image (OpenCV format) and runs end-to-end segmentation
        & passability analysis. High-throughput inference mode.

        Returns:
            pred_mask_orig (H, W),
            overlay_bgr (H, W, 3),
            depth_heatmap_bgr (H, W, 3) or None,
            passability_info (dict)
        """
        h_orig, w_orig, _ = bgr_img.shape

        # Fast preprocessing without unnecessary memory copy
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb_img, self.img_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if self.device.type == 'cuda':
            img_tensor = img_tensor.to(self.device, non_blocking=True)

        # High-Speed Model Inference (FP16 on GPU for ~1.5× speedup)
        start_t = time.perf_counter()
        with torch.inference_mode():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = self.model(img_tensor)
            else:
                logits = self.model(img_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        # Resize prediction mask back to original resolution
        pred_mask_orig = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Passability Analysis — choose backend
        if self.depth_backend == 'sturm' and self._sturm_estimator is not None:
            passability_info = self.analyzer.analyze_with_sturm(
                pred_mask_orig,
                bgr_img,
                sturm_estimator=self._sturm_estimator,
                colormap=self.colormap,
                generate_heatmap=generate_heatmap,
                sturm_telemetry=sturm_telemetry,
            )
        else:
            passability_info = self.analyzer.analyze(
                pred_mask_orig,
                colormap=self.colormap,
                generate_heatmap=generate_heatmap,
            )

        passability_info['latency_ms'] = round(latency_ms, 2)
        passability_info['fps'] = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0

        # Generate HUD Visual Overlay
        overlay_bgr = self.analyzer.generate_visual_overlay(bgr_img, pred_mask_orig, passability_info)

        # Retrieve Depth Heatmap if requested
        depth_heatmap_bgr = passability_info.get('_depth_heatmap_bgr')
        if generate_heatmap and depth_heatmap_bgr is None:
            depth_heatmap_bgr = self.analyzer.generate_depth_heatmap(
                pred_mask_orig, passability_info, original_bgr=bgr_img, colormap=self.colormap
            )

        return pred_mask_orig, overlay_bgr, depth_heatmap_bgr, passability_info


def main():
    parser = argparse.ArgumentParser(description="Run Flood Segmentation & Road Passability Inference")
    parser.add_argument('--input',              type=str, required=True,       help='Path to input image file')
    parser.add_argument('--model',              type=str, default='mobilenet_unet', choices=['unet', 'mobilenet_unet'])
    parser.add_argument('--weights',            type=str, default=None,        help='Path to model weights checkpoint')
    parser.add_argument('--out',                type=str, default='outputs/prediction.jpg', help='Output visualization path')
    parser.add_argument('--depth-backend',      type=str, default='sturm', choices=['sturm', 'morphological'],
                        help='Depth backend: sturm (Zenodo 14833532, recommended) or morphological')
    parser.add_argument('--sturm-weights',      type=str, default=None,        help='Path to STURM flood level weights checkpoint')
    parser.add_argument('--colormap',           type=str, default='turbo',     choices=['inferno', 'turbo', 'magma', 'plasma', 'viridis'],
                        help='Depth colormap (turbo is signature Bathymetric/STURM)')
    parser.add_argument('--depth-heatmap-out',  type=str, default=None,
                        help='Path to save Depth heatmap (default: outputs/depth_heatmap.jpg)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    # Check if input is a video file
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    if args.input.lower().endswith(video_exts):
        print(f"[Video Mode] Detected video file: {args.input}")
        from inference_video import process_video
        output_video = args.out if args.out.lower().endswith(video_exts) else "outputs/annotated_video.mp4"
        process_video(
            input_path=args.input,
            output_path=output_video,
            depth_backend=args.depth_backend,
            model_type=args.model,
            stride=2,
            scale=1.0,
        )
        return

    bgr_img = cv2.imread(args.input)
    predictor = FloodSegmentationPredictor(
        model_type=args.model,
        weights_path=args.weights,
        depth_backend=args.depth_backend,
        sturm_weights=args.sturm_weights,
        colormap=args.colormap,
    )

    pred_mask, overlay, depth_heatmap, summary = predictor.predict_image(bgr_img)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cv2.imwrite(args.out, overlay)

    depth_out_path = args.depth_heatmap_out
    if depth_out_path is None:
        depth_out_path = os.path.join(os.path.dirname(args.out) or 'outputs', f"{args.depth_backend}_depth_heatmap.jpg")

    if depth_out_path and depth_heatmap is not None:
        os.makedirs(os.path.dirname(depth_out_path) or '.', exist_ok=True)
        cv2.imwrite(depth_out_path, depth_heatmap)

    print("\n" + "=" * 62)
    print("      DUAL-STREAM MULTI-MODAL PASSABILITY TELEMETRY")
    print("=" * 62)
    print(f"Input Scene Image    : {args.input}")
    print(f"Inference Latency    : {summary['latency_ms']} ms ({summary['fps']} FPS)")

    # Stream 1 - Spatial 2D Coverage
    s1 = summary.get('stream1_spatial', {})
    print("\n[STREAM 1: 2D ROAD & FLOOD SEGMENTATION (MobileNet-UNet)]")
    print(f"  * Flooded Road Ratio : {summary['flooded_road_ratio']}%")
    print(f"  * Roadbed Continuity : {s1.get('road_continuity', 'Continuous')}")
    print(f"  * Active Segments    : {summary.get('road_segments_count', 1)}")

    # Stream 2 - Vertical Depth
    s2 = summary.get('stream2_depth', {})
    backend_name = s2.get('backend', summary.get('depth_source', 'STURM-FloodDepth'))
    print(f"\n[STREAM 2: DEPTH ESTIMATION ({backend_name})]")
    if summary.get('sturm_level') is not None:
        print(f"  * STURM Flood Level  : {summary.get('sturm_level_label')} ({summary.get('sturm_level_title')})")
        print(f"  * Confidence         : {summary.get('sturm_confidence_pct')}%")
        print(f"  * Depth Range        : {summary.get('sturm_depth_range_str')}")
        print(f"  * Vehicle Reference  : {summary.get('sturm_landmark')}")
        print(f"  * Hazard Assessment  : {summary.get('sturm_hazard')}")
    print(f"  * Calibrated Peak    : {summary['max_depth_cm']} cm")
    print(f"  * Calibrated Mean    : {summary['mean_depth_cm']} cm")

    # Multi-Modal Passability Verdict
    print("\n[MULTI-MODAL PASSABILITY FUSION DECISION]")
    print(f"  * Overall Status     : {summary['status'].upper()}")
    print(f"  * Decisive Factor    : {summary.get('primary_factor', 'Dual-Stream Verified')}")
    print(f"  * Driving Advisory   : {summary['recommendation']}")

    print("\n[SAVED VISUAL ARTIFACTS]")
    print(f"  * HUD Passability Overlay : {args.out}")
    if depth_out_path and depth_heatmap is not None:
        print(f"  * Depth Heatmap Artifact : {depth_out_path}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
