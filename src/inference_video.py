"""
Video Inference Pipeline for Flood Segmentation & Road Passability Estimation
=============================================================================
Processes street/dashcam video streams frame-by-frame:
1. Real-time Semantic Segmentation (MobileNet-UNet) for 2D road & flood coverage.
2. STURM-FloodDepth (MobileNetV3) for physical vertical flood depth levels (Level 0–4).
3. Connected road geometry and passability classification.
4. Generates an annotated video with a HUD overlay, timeline depth graph, and JSON telemetry.

Usage:
    python src/inference_video.py --input data/test_video.mp4 --depth-backend sturm
    python src/inference_video.py --input data/test_video.mp4 --stride 2 --scale 0.5
"""

import os
import sys

# Ensure matplotlib cache directory is in workspace & use non-interactive Agg backend
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("outputs"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import time
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

from inference import FloodSegmentationPredictor


def process_video(
    input_path: str,
    output_path: str = "outputs/annotated_video.mp4",
    depth_backend: str = "sturm",
    model_type: str = "mobilenet_unet",
    stride: int = 2,
    scale: float = 1.0,
    max_frames: int | None = None,
    plot_summary: bool = True,
    progress_callback = None,
    predictor = None,
    keyframe_interval: int = 2,
):
    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' not found.")
        return None

    print("=" * 65)
    print("      VIDEO FLOOD SEGMENTATION & ROAD PASSABILITY PIPELINE")
    print("=" * 65)
    print(f"Input Video    : {input_path}")
    print(f"Output Video   : {output_path}")
    print(f"Depth Backend  : {depth_backend.upper()}")
    print(f"Frame Stride   : {stride} (Processing 1 of every {stride} frames)")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return None

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / orig_fps if orig_fps > 0 else 0

    target_w = int(orig_w * scale)
    target_h = int(orig_h * scale)
    # Ensure dimensions are even for video codecs
    target_w = target_w if target_w % 2 == 0 else target_w - 1
    target_h = target_h if target_h % 2 == 0 else target_h - 1
    output_fps = max(1.0, orig_fps / stride)

    print(f"Source Specs   : {orig_w}x{orig_h} @ {orig_fps:.1f} FPS ({total_frames} frames, {duration_sec:.1f}s)")
    print(f"Target Specs   : {target_w}x{target_h} @ {output_fps:.1f} FPS")

    # Initialize predictor if not provided
    if predictor is None:
        predictor = FloodSegmentationPredictor(
            model_type=model_type,
            depth_backend=depth_backend,
            colormap="turbo",
        )

    # Initialize VideoWriter (prefer browser-compatible H.264 via Windows Media Foundation)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_writer = cv2.VideoWriter(output_path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), output_fps, (target_w, target_h))
    if not out_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (target_w, target_h))

    telemetry_timeline = []
    frame_idx = 0
    processed_count = 0
    t_start = time.time()

    # STURM caching & Keyframe mask reuse for real-time 15-20+ FPS
    cached_sturm = None
    sturm_interval = 10  # run STURM every 10th processed frame
    last_mask = None
    last_telemetry = None

    total_to_process = (total_frames // stride) if max_frames is None else min(max_frames, total_frames // stride)
    total_to_process = max(1, total_to_process)
    pbar = tqdm(total=total_to_process, desc="Processing Video", unit="frame")

    while cap.isOpened():
        # Fast skip non-stride frames without expensive JPEG/H264 decoding
        if frame_idx % stride != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:
            frame_input = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_input = frame

        # Keyframe optimization: run full segmentation model every Nth frame,
        # propagate previous mask on intermediate frames for ultra-smooth 15-20+ FPS
        is_keyframe = (keyframe_interval <= 1) or (processed_count % keyframe_interval == 0) or (last_mask is None)

        if is_keyframe:
            use_sturm = cached_sturm if (processed_count % sturm_interval != 0) else None
            last_mask, overlay, _, telemetry = predictor.predict_image(
                frame_input, generate_heatmap=False, sturm_telemetry=use_sturm
            )
            fresh_sturm = telemetry.get('_sturm_telemetry')
            if fresh_sturm is not None:
                cached_sturm = fresh_sturm
            last_telemetry = telemetry
        else:
            telemetry = last_telemetry.copy()
            overlay = predictor.analyzer.generate_visual_overlay(frame_input, last_mask, telemetry)

        # Embed Timeline / Timestamp watermark on HUD overlay
        timestamp_sec = frame_idx / orig_fps
        mins = int(timestamp_sec // 60)
        secs = int(timestamp_sec % 60)
        ms = int((timestamp_sec - int(timestamp_sec)) * 10)
        time_str = f"Time: {mins:02d}:{secs:02d}.{ms} / {int(duration_sec//60):02d}:{int(duration_sec%60):02d}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.42, target_h / 1400.0)
        cv2.putText(overlay, time_str, (20, target_h - 15), font, fs, (220, 220, 220), 1, cv2.LINE_AA)

        # Record telemetry
        telemetry_point = {
            'frame_idx': frame_idx,
            'timestamp_sec': round(timestamp_sec, 2),
            'status': telemetry.get('status', 'Passable'),
            'flooded_road_ratio': telemetry.get('flooded_road_ratio', 0.0),
            'max_depth_cm': telemetry.get('max_depth_cm', 0.0),
            'mean_depth_cm': telemetry.get('mean_depth_cm', 0.0),
            'sturm_level': telemetry.get('sturm_level', 0),
            'sturm_level_label': telemetry.get('sturm_level_label', 'Level 0'),
            'sturm_confidence_pct': telemetry.get('sturm_confidence_pct', 0.0),
            'sturm_landmark': telemetry.get('sturm_landmark', ''),
            'recommendation': telemetry.get('recommendation', ''),
        }
        telemetry_timeline.append(telemetry_point)

        out_writer.write(overlay)
        processed_count += 1
        pbar.update(1)

        if progress_callback:
            pct = min(99.0, round((processed_count / total_to_process) * 100.0, 1))
            cur_fps = round(processed_count / (time.time() - t_start), 1) if (time.time() - t_start) > 0 else 0.0
            status_msg = f"Processing frame {processed_count}/{total_to_process} ({cur_fps} FPS)"
            live_telemetry = {
                'fps': cur_fps,
                'depth_cm': telemetry_point.get('max_depth_cm', 0.0),
                'ratio_pct': telemetry_point.get('flooded_road_ratio', 0.0),
                'status': telemetry_point.get('status', 'Passable'),
                'sturm_level': telemetry_point.get('sturm_level', 0),
                'sturm_label': telemetry_point.get('sturm_level_label', 'Level 0'),
                'sturm_landmark': telemetry_point.get('sturm_landmark', ''),
                'recommendation': telemetry_point.get('recommendation', ''),
            }
            try:
                import inspect
                sig = inspect.signature(progress_callback)
                if len(sig.parameters) >= 5:
                    progress_callback(pct, processed_count, total_to_process, status_msg, live_telemetry)
                else:
                    progress_callback(pct, processed_count, total_to_process, status_msg)
            except Exception:
                try:
                    progress_callback(pct, processed_count, total_to_process, status_msg)
                except Exception:
                    pass

        if max_frames and processed_count >= max_frames:
            break

        frame_idx += 1

    pbar.close()
    cap.release()
    out_writer.release()
    total_elapsed = time.time() - t_start

    fps_achieved = processed_count / total_elapsed if total_elapsed > 0 else 0
    print(f"\nVideo processing completed in {total_elapsed:.1f}s ({fps_achieved:.1f} FPS processing speed)")
    print(f"Saved annotated video to: {output_path}")

    # Summary Statistics
    summary = None
    if telemetry_timeline:
        statuses = [p['status'] for p in telemetry_timeline]
        depths = [p['max_depth_cm'] for p in telemetry_timeline]
        ratios = [p['flooded_road_ratio'] for p in telemetry_timeline]
        sturm_levels = [p['sturm_level'] for p in telemetry_timeline]

        n_total = len(statuses)
        passable_pct = (statuses.count('Passable') / n_total) * 100
        partial_pct = (statuses.count('Partially Flooded') / n_total) * 100
        severe_pct = ((statuses.count('Severely Flooded') + statuses.count('Severely Affected')) / n_total) * 100

        peak_depth = max(depths) if depths else 0.0
        peak_ratio = max(ratios) if ratios else 0.0
        max_level = max(sturm_levels) if sturm_levels else 0

        summary = {
            'input_video': input_path,
            'output_video': output_path,
            'duration_sec': round(duration_sec, 2),
            'frames_processed': processed_count,
            'average_processing_fps': round(fps_achieved, 1),
            'passability_distribution': {
                'passable_pct': round(passable_pct, 1),
                'partially_flooded_pct': round(partial_pct, 1),
                'severely_flooded_pct': round(severe_pct, 1),
            },
            'peak_flood_depth_cm': round(peak_depth, 1),
            'peak_flooded_ratio_pct': round(peak_ratio, 1),
            'peak_sturm_level': max_level,
            'timeline': telemetry_timeline,
        }

        # Save JSON Report
        json_out = os.path.splitext(output_path)[0] + "_telemetry.json"
        with open(json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved telemetry summary to: {json_out}")

        # Generate Timeline Plot
        if plot_summary:
            plot_video_telemetry(telemetry_timeline, os.path.splitext(output_path)[0] + "_profile.png")

        print("\n" + "=" * 65)
        print("                 VIDEO JOURNEY SUMMARY REPORT")
        print("=" * 65)
        print(f"Peak Flood Depth Encountered : {peak_depth:.1f} cm (STURM Level {max_level})")
        print(f"Peak Flooded Surface Ratio   : {peak_ratio:.1f}%")
        print(f"Journey Passability Breakdown:")
        print(f"  * 🟢 Passable Safe Driving   : {passable_pct:.1f}% of journey")
        print(f"  * 🟡 Partially Flooded (SUVs): {partial_pct:.1f}% of journey")
        print(f"  * 🔴 Severely Flooded Hazard : {severe_pct:.1f}% of journey")
        print("=" * 65 + "\n")

    if progress_callback:
        try:
            progress_callback(100.0, processed_count, total_to_process, "Video processing completed!")
        except Exception:
            pass

    return summary


def plot_video_telemetry(timeline: list[dict], save_path: str):
    """Plots temporal depth profile and passability zones across the video journey."""
    timestamps = [p['timestamp_sec'] for p in timeline]
    depths = [p['max_depth_cm'] for p in timeline]
    ratios = [p['flooded_road_ratio'] for p in timeline]
    levels = [p['sturm_level'] for p in timeline]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # 1. Depth Profile Plot
    ax1.plot(timestamps, depths, color='#0284c7', linewidth=2, label='Est. Peak Water Depth (cm)')
    ax1.axhline(15.0, color='#10b981', linestyle='--', alpha=0.7, label='Passable Threshold (15 cm)')
    ax1.axhline(30.0, color='#f59e0b', linestyle='--', alpha=0.7, label='SUV Clearance Limit (30 cm)')
    ax1.axhline(50.0, color='#ef4444', linestyle='--', alpha=0.7, label='Critical Hydrolock Hazard (50 cm)')
    ax1.fill_between(timestamps, 0, depths, color='#0284c7', alpha=0.25)
    ax1.set_ylabel('Water Depth (cm)', fontsize=11, fontweight='bold')
    ax1.set_title('Temporal Flood Depth & Hazard Profile Across Video Stream', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # 2. STURM Flood Level & Road Coverage Ratio
    ax2.plot(timestamps, ratios, color='#f97316', linewidth=2, label='Flooded Roadbed Ratio (%)')
    ax2_twin = ax2.twinx()
    ax2_twin.step(timestamps, levels, color='#8b5cf6', linewidth=2, where='mid', label='STURM Level (0 to 4)')
    ax2_twin.set_ylabel('STURM Level (0-4)', color='#8b5cf6', fontsize=11, fontweight='bold')
    ax2_twin.set_yticks([0, 1, 2, 3, 4])
    ax2_twin.set_yticklabels(['L0 (Dry)', 'L1 (<25cm)', 'L2 (25-40cm)', 'L3 (40-60cm)', 'L4 (>60cm)'])
    
    ax2.set_xlabel('Video Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Flooded Surface Ratio (%)', color='#f97316', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved temporal depth profile chart to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Video Inference for Flood Segmentation & Passability")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file (.mp4, .avi, etc.)")
    parser.add_argument("--output", type=str, default="outputs/annotated_video.mp4", help="Path to save output video")
    parser.add_argument("--depth-backend", type=str, default="sturm", choices=["sturm", "morphological"],
                        help="Depth backend engine (default: sturm)")
    parser.add_argument("--model", type=str, default="mobilenet_unet", choices=["unet", "mobilenet_unet"],
                        help="Segmentation model architecture")
    parser.add_argument("--stride", type=int, default=2, help="Frame sampling stride (e.g. 2 = process every 2nd frame)")
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scaling factor (e.g. 0.5 for 2x faster processing)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process (optional)")
    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        depth_backend=args.depth_backend,
        model_type=args.model,
        stride=args.stride,
        scale=args.scale,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
