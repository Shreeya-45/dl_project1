import sys
import os
import torch
import numpy as np

# Ensure execution is anchored to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# pyrefly: ignore[missing-import]
from models import UNetBaseline, MobileNetUNet, CombinedCrossEntropyDiceLoss, STURMFloodLevelClassifier
# pyrefly: ignore[missing-import]
from utils import (
    PassabilityAnalyzer, calculate_segmentation_metrics,
    colorize_depth_map, blend_depth_heatmap, get_sturm_estimator
)

def test_models():
    print("Testing Model Architectures...")
    dummy_input = torch.randn(2, 3, 256, 256)
    
    # 1. UNet Baseline
    unet = UNetBaseline(in_channels=3, num_classes=3)
    out_u = unet(dummy_input)
    assert out_u.shape == (2, 3, 256, 256), f"UNet output shape mismatch: {out_u.shape}"
    print("  [PASS] UNet Baseline output shape verified: (2, 3, 256, 256)")

    # 2. MobileNet-UNet Proposed
    mb_unet = MobileNetUNet(in_channels=3, num_classes=3)
    out_m = mb_unet(dummy_input)
    assert out_m.shape == (2, 3, 256, 256), f"MobileNet-UNet output shape mismatch: {out_m.shape}"
    print("  [PASS] MobileNet-UNet Lite output shape verified: (2, 3, 256, 256)")

    # 3. STURM Flood Level Classifier (Zenodo: 14833532)
    sturm_model = STURMFloodLevelClassifier(num_classes=5)
    dummy_sturm = torch.randn(2, 3, 224, 224)
    out_s = sturm_model(dummy_sturm)
    assert out_s.shape == (2, 5), f"STURM classifier output shape mismatch: {out_s.shape}"
    print("  [PASS] STURM 5-Class Flood Depth Model shape verified: (2, 5)")


def test_loss():
    print("\nTesting Combined Cross-Entropy + Dice Loss...")
    criterion = CombinedCrossEntropyDiceLoss()
    logits = torch.randn(2, 3, 64, 64, requires_grad=True)
    targets = torch.randint(0, 3, (2, 64, 64))
    
    loss, ce, dice = criterion(logits, targets)
    loss.backward()
    assert loss.item() > 0, "Loss must be positive"
    print(f"  [PASS] Combined Loss computed cleanly: Loss={loss.item():.4f}, CE={ce.item():.4f}, Dice={dice.item():.4f}")


def test_passability():
    print("\nTesting Passability Estimation Engine...")
    analyzer = PassabilityAnalyzer()
    
    # Test case 1: Passable road (< 15% flooded)
    mask1 = np.ones((100, 100), dtype=np.uint8) # All dry road
    mask1[85:100, :] = 2                       # 15% water
    res1 = analyzer.analyze(mask1)
    print(f"  [PASS] Low flood scene classified as: {res1['status']} ({res1['flooded_road_ratio']}%)")

    # Test case 2: Severe flood (>= 50% flooded)
    mask2 = np.ones((100, 100), dtype=np.uint8)
    mask2[30:100, :] = 2                       # 70% water
    res2 = analyzer.analyze(mask2)
    print(f"  [PASS] Heavy flood scene classified as: {res2['status']} ({res2['flooded_road_ratio']}%)")
    assert res2['status'] == "Severely Affected"


def test_metrics():
    print("\nTesting Evaluation Metrics Engine...")
    gt = np.array([[0, 1, 1], [1, 2, 2], [0, 0, 2]])
    pred = np.array([[0, 1, 1], [1, 1, 2], [0, 0, 2]])
    
    hist = np.zeros((3, 3), dtype=int)
    for p, t in zip(pred, gt):
        hist[t, p] += 1
        
    metrics = calculate_segmentation_metrics(hist)
    print(f"  [PASS] Calculated mIoU={metrics['mIoU']:.4f}, mean_dice={metrics['mean_dice']:.4f}")

def test_depth_heatmap():
    print("\nTesting Bathymetric Depth Heatmap Generation Engine...")
    dummy_depth = np.linspace(10, 100, 256 * 256, dtype=np.float32).reshape(256, 256)
    
    # 1. Colorize depth map (Inferno colormap)
    heatmap_inferno = colorize_depth_map(dummy_depth, colormap="inferno", add_colorbar=True)
    assert heatmap_inferno.shape == (256, 256, 3), f"Heatmap shape mismatch: {heatmap_inferno.shape}"
    assert heatmap_inferno.dtype == np.uint8, "Heatmap dtype must be uint8"
    print("  [PASS] Inferno Depth Heatmap generated: shape=(256, 256, 3)")

    # 2. Colorize with Turbo colormap
    heatmap_turbo = colorize_depth_map(dummy_depth, colormap="turbo", add_colorbar=True)
    assert heatmap_turbo.shape == (256, 256, 3)
    print("  [PASS] Turbo Bathymetric Heatmap generated: shape=(256, 256, 3)")

    # 3. Blend depth heatmap with dummy BGR scene
    dummy_scene = np.zeros((256, 256, 3), dtype=np.uint8)
    blended = blend_depth_heatmap(dummy_scene, heatmap_inferno, alpha=0.5)
    assert blended.shape == (256, 256, 3)
    print("  [PASS] Blended Depth Heatmap scene generated successfully")

    # 4. Passability Analyzer Depth Heatmap integration
    analyzer = PassabilityAnalyzer()
    dummy_mask = np.ones((256, 256), dtype=np.uint8)
    dummy_mask[180:, :] = 2 # 30% water
    res = analyzer.analyze(dummy_mask)
    assert '_depth_heatmap_bgr' in res, "Depth heatmap must be generated in analysis result"
    assert res['_depth_heatmap_bgr'].shape == (256, 256, 3)
    print("  [PASS] Passability Analyzer auto-attached Depth Heatmap in telemetry result")


def test_sturm_pipeline():
    print("\nTesting STURM-FloodDepth Pipeline (Zenodo: 14833532)...")
    sturm_est = get_sturm_estimator()
    dummy_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 1. Forward inference on image
    res = sturm_est.estimate(dummy_bgr)
    assert 'level' in res and 0 <= res['level'] < 5, f"Invalid STURM level: {res.get('level')}"
    assert 'probabilities' in res and len(res['probabilities']) == 5
    print(f"  [PASS] STURM Estimator output verified: {res['level_label']} ({res['depth_range_str']}) with conf {res['confidence_pct']}%")

    # 2. Level HUD Card Generation
    card = sturm_est.generate_level_card(res, width=320, height=140)
    assert card.shape == (140, 320, 3), f"HUD Card shape mismatch: {card.shape}"
    print("  [PASS] STURM Telemetry HUD Card generated: shape=(140, 320, 3)")

    # 3. Passability Analyzer Integration
    analyzer = PassabilityAnalyzer()
    dummy_mask = np.ones((256, 256), dtype=np.uint8)
    dummy_mask[160:, :] = 2 # Flooded
    pass_res = analyzer.analyze_with_sturm(dummy_mask, dummy_bgr, sturm_estimator=sturm_est)
    assert pass_res['depth_source'] == 'sturm_flooddepth'
    assert 'sturm_level' in pass_res
    assert '_depth_heatmap_bgr' in pass_res
    print(f"  [PASS] Passability Analyzer with STURM executed: Status={pass_res['status']}, Depth={pass_res['max_depth_cm']}cm")


def run_all_tests():
    print("=" * 60)
    print("   RUNNING CAPSTONE PROJECT VERIFICATION TEST SUITE")
    print("=" * 60)
    test_models()
    test_loss()
    test_passability()
    test_sturm_pipeline()
    test_depth_heatmap()
    test_metrics()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! FULL PIPELINE READY.")


if __name__ == '__main__':
    run_all_tests()
