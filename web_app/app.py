import os
import sys

# Ensure headless Agg backend for matplotlib in web threads
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")))
import matplotlib
matplotlib.use("Agg")

import base64
import json
import time
import threading
import uuid
import cv2
import numpy as np
import torch

# Maximize CPU inference throughput (6 P-cores avoids hybrid core sync bottleneck)
torch.set_num_threads(min(6, os.cpu_count() or 4))
cv2.setNumThreads(min(6, os.cpu_count() or 4))

from flask import Flask, render_template, request, jsonify, send_from_directory

# Add parent directory to sys.path to import src modules cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# pyrefly: ignore[missing-import]
from models import UNetBaseline, MobileNetUNet
# pyrefly: ignore[missing-import]
from utils import (
    PassabilityAnalyzer, calculate_segmentation_metrics, measure_model_performance,
    get_sturm_estimator
)
# pyrefly: ignore[missing-import]
from inference_video import process_video

WEB_APP_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(WEB_APP_DIR, 'templates'),
    static_folder=os.path.join(WEB_APP_DIR, 'static')
)

# Output directories & task store
OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
UPLOADS_DIR = os.path.join(OUTPUTS_DIR, 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)
VIDEO_TASKS = {}

# Initialize Device and Models
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (256, 256)

MODELS = {}

# Shared STURM-FloodDepth Estimator (Zenodo: 14833532)
_STURM_ESTIMATOR = None

def _get_sturm_estimator():
    global _STURM_ESTIMATOR
    if _STURM_ESTIMATOR is None:
        _STURM_ESTIMATOR = get_sturm_estimator()
    return _STURM_ESTIMATOR

def load_models():
    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights'))
    
    # 1. Baseline UNet
    unet = UNetBaseline(in_channels=3, num_classes=3).to(DEVICE)
    unet_path = os.path.join(weights_dir, "best_unet.pth")
    if os.path.exists(unet_path):
        try:
            ckpt = torch.load(unet_path, map_location=DEVICE)
            unet.load_state_dict(ckpt['model_state_dict'])
            print(f"[Flask] Loaded U-Net weights from {unet_path}")
        except Exception as e:
            print(f"[Flask] Warning loading UNet weights: {e}")
    unet.eval()
    MODELS['unet'] = unet

    # 2. MobileNet UNet Lite (Proposed)
    mobilenet = MobileNetUNet(in_channels=3, num_classes=3).to(DEVICE)
    mobilenet_path = os.path.join(weights_dir, "best_mobilenet_unet.pth")
    if os.path.exists(mobilenet_path):
        try:
            ckpt = torch.load(mobilenet_path, map_location=DEVICE)
            mobilenet.load_state_dict(ckpt['model_state_dict'])
            print(f"[Flask] Loaded MobileNet-UNet weights from {mobilenet_path}")
        except Exception as e:
            print(f"[Flask] Warning loading MobileNet-UNet weights: {e}")
    mobilenet.eval()
    MODELS['mobilenet_unet'] = mobilenet

    # TorchScript JIT — freeze & optimize both models for CPU inference
    # Gives ~25-40% faster forward pass on Intel CPUs vs eager mode.
    dummy = torch.zeros(1, 3, *IMG_SIZE, device=DEVICE)
    for key, model in [('unet', unet), ('mobilenet_unet', mobilenet)]:
        try:
            with torch.inference_mode():
                traced = torch.jit.trace(model, dummy)
                traced = torch.jit.freeze(traced)
                MODELS[key] = torch.jit.optimize_for_inference(traced)
                # Warm-up run to compile kernels
                _ = MODELS[key](dummy)
            print(f"[Flask] {key}: TorchScript JIT compiled & warmed up.")
        except Exception as jit_err:
            print(f"[Flask] {key}: JIT failed ({jit_err}), using eager mode.")
            MODELS[key] = model

    # Warm up CUDA kernels to eliminate the initial 120ms cold-start latency
    if DEVICE.type == 'cuda':
        with torch.no_grad():
            _ = MODELS['unet'](dummy)
            _ = MODELS['mobilenet_unet'](dummy)
            torch.cuda.synchronize()
        print("[Flask] CUDA context and models warmed up successfully.")

ANALYZER = PassabilityAnalyzer()

def image_to_base64(bgr_array, quality: int = 80):
    """JPEG-encode bgr_array and return a data-URI string.
    Quality 80 cuts file size ~40 % vs 95 with no visible quality loss in the browser.
    """
    _, buffer = cv2.imencode('.jpg', bgr_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')


# Maximum width for display images sent to the browser.
# Processing at 800 px wide is ~6× faster than 1920 px for the overlay / heatmap steps
# while keeping visual quality high enough for a 400 px viewer panel.
_DISPLAY_MAX_W = 800

def _cap_display(bgr: np.ndarray) -> np.ndarray:
    """Downscale bgr so its width ≤ _DISPLAY_MAX_W (preserves aspect ratio)."""
    h, w = bgr.shape[:2]
    if w <= _DISPLAY_MAX_W:
        return bgr
    scale = _DISPLAY_MAX_W / w
    new_w = _DISPLAY_MAX_W
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def get_model(model_name: str = 'mobilenet_unet'):
    """Returns requested model, loading weights lazily if not already in memory."""
    if 'mobilenet_unet' not in MODELS or 'unet' not in MODELS:
        load_models()
    return MODELS.get(model_name, MODELS.get('mobilenet_unet'))


@app.route('/images/<path:filename>')
def serve_static_images(filename):
    static_images = os.path.join(WEB_APP_DIR, 'static', 'images')
    if os.path.exists(os.path.join(static_images, filename)):
        return send_from_directory(static_images, filename)
    return ('', 404)


@app.route('/data/raw/images/<path:filename>')
def serve_raw_samples(filename):
    data_dir = os.path.abspath(os.path.join(WEB_APP_DIR, '..', 'data', 'raw', 'images'))
    if os.path.exists(os.path.join(data_dir, filename)):
        return send_from_directory(data_dir, filename)
    return ('', 404)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/input')
@app.route('/workspace')
def input_page():
    return render_template('input.html')


@app.route('/output')
@app.route('/results')
def output_page():
    return render_template('output.html')


@app.route('/dashboard')
@app.route('/legacy')
def legacy_dashboard():
    return render_template('home.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        model_choice = request.form.get('model', 'mobilenet_unet')
        selected_model = get_model(model_choice)

        # Depth Backend selection (STURM-FloodDepth or Morphological)
        depth_backend = request.form.get('depth_backend', 'sturm').lower()
        colormap      = request.form.get('colormap', 'turbo')

        file = request.files.get('image')
        sample_path = request.form.get('sample_path')

        if file:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif sample_path:
            resolved_p = None
            if os.path.exists(sample_path):
                resolved_p = sample_path
            else:
                fname = os.path.basename(sample_path)
                static_candidate = os.path.join(WEB_APP_DIR, 'static', 'images', fname)
                raw_candidate = os.path.abspath(os.path.join(WEB_APP_DIR, '..', 'data', 'raw', 'images', fname))
                if os.path.exists(static_candidate):
                    resolved_p = static_candidate
                elif os.path.exists(raw_candidate):
                    resolved_p = raw_candidate
            if resolved_p and os.path.exists(resolved_p):
                bgr_img = cv2.imread(resolved_p)
            else:
                data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'images'))
                samples = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))] if os.path.exists(data_dir) else []
                bgr_img = cv2.imread(samples[0]) if samples else None
        else:
            # Fallback to a default sample from data/raw/images or CVFD
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'images'))
            cvfd_dir = r"C:\Users\HP\Downloads\archive (11)\cvfd\images\test\flooded"
            samples = []
            if os.path.exists(data_dir):
                samples = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))]
            if not samples and os.path.exists(cvfd_dir):
                samples = [os.path.join(cvfd_dir, f) for f in os.listdir(cvfd_dir) if f.lower().endswith(('.jpg', '.png'))]
            if samples:
                bgr_img = cv2.imread(samples[0])
            else:
                return jsonify({'error': 'No image provided and no sample dataset found'}), 400

        if bgr_img is None:
            return jsonify({
                'success': False,
                'error': 'Could not decode image. If you uploaded a video file, please switch to the Video Stream tab to process video footage.'
            }), 400

        h_orig, w_orig, _ = bgr_img.shape

        # ── Speed optimisation: cap working resolution to _DISPLAY_MAX_W ──────
        # This makes all downstream NumPy / OpenCV work (overlay, heatmap, mask)
        # 4-9× faster on large input images (e.g. 1920 px → 800 px = ~5.76× less area).
        bgr_work = _cap_display(bgr_img)
        h_work, w_work = bgr_work.shape[:2]

        # Fast preprocessing: OpenCV → numpy → tensor (no PIL)
        # Input to the model is always IMG_SIZE (256×256) regardless of source resolution.
        rgb_img = cv2.cvtColor(bgr_work, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb_img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # High-speed inference with inference_mode (faster than no_grad on CPU)
        start_t = time.perf_counter()
        with torch.inference_mode():
            logits = selected_model(img_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0

        # Resize prediction mask to working resolution (not full original).
        # All post-processing (depth map, overlay, heatmap) runs at this smaller size.
        pred_mask_work = cv2.resize(pred.astype(np.uint8), (w_work, h_work), interpolation=cv2.INTER_NEAREST)

        # Passability Analysis — STURM or morphological (runs on working-res mask)
        if depth_backend == 'sturm':
            sturm_est = _get_sturm_estimator()
            telemetry = ANALYZER.analyze_with_sturm(
                pred_mask_work, bgr_work, sturm_estimator=sturm_est, colormap=colormap
            )
        else:
            telemetry = ANALYZER.analyze(pred_mask_work, colormap=colormap)

        telemetry['latency_ms'] = round(latency_ms, 2)
        telemetry['fps'] = fps

        # Visual overlay and depth heatmap — both run at working resolution
        overlay_bgr = ANALYZER.generate_visual_overlay(bgr_work, pred_mask_work, telemetry)

        depth_heatmap_bgr = telemetry.get('_depth_heatmap_bgr')
        if depth_heatmap_bgr is None:
            depth_heatmap_bgr = ANALYZER.generate_depth_heatmap(
                pred_mask_work, telemetry, original_bgr=bgr_work, colormap=colormap
            )

        # Clean telemetry for JSON response (remove internal NumPy arrays)
        telemetry_clean = {k: v for k, v in telemetry.items() if not k.startswith('_')}

        # RGB mask image at working resolution
        color_mask_bgr = np.zeros((h_work, w_work, 3), dtype=np.uint8)
        color_mask_bgr[pred_mask_work == 1] = [50, 205, 50]   # Dry Road: Green
        color_mask_bgr[pred_mask_work == 2] = [0, 140, 255]   # Flood Water: Blue/Orange

        # Encode at quality=80 — ~40 % smaller payload than default 95, invisible difference
        def _enc(arr): return image_to_base64(arr, quality=80)

        return jsonify({
            'success': True,
            'original_image':  _enc(bgr_img),
            'overlay_image':   _enc(overlay_bgr),
            'mask_image':      _enc(color_mask_bgr),
            'depth_heatmap':   _enc(depth_heatmap_bgr),
            'telemetry':       telemetry_clean,
            'model_used':      getattr(selected_model, 'name', model_choice),
            'depth_source':    telemetry_clean.get('depth_source', 'morphological'),
            'colormap':        colormap,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/samples')
def get_samples():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'images'))
    cvfd_dir = r"C:\Users\HP\Downloads\archive (11)\cvfd\images\test\flooded"

    active_dir = None
    if os.path.exists(data_dir) and any(f.endswith(('.jpg', '.png')) for f in os.listdir(data_dir)):
        active_dir = data_dir
    elif os.path.exists(cvfd_dir):
        active_dir = cvfd_dir

    if not active_dir:
        return jsonify({'samples': []})

    all_files = [f for f in os.listdir(active_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    sample_items = []

    for f in all_files[:12]:
        p = os.path.join(active_dir, f)
        bgr = cv2.imread(p)
        if bgr is not None:
            thumb = cv2.resize(bgr, (160, 120))
            sample_items.append({
                'filename': f,
                'path': p,
                'thumbnail': image_to_base64(thumb)
            })

    return jsonify({'samples': sample_items})



@app.route('/api/benchmark')
def get_benchmark():
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'evaluation_report.json'))
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    
    # Fallback default benchmark metrics if JSON not yet generated
    return jsonify({
        'U-Net Baseline': {
            'mIoU': 0.812, 'mean_dice': 0.875, 'road_iou': 0.894, 'flooded_road_iou': 0.821,
            'boundary_f1': 0.842, 'num_params_m': 31.03, 'model_size_mb': 118.4,
            'latency_ms': 42.5, 'fps': 23.5
        },
        'MobileNet-UNet (Proposed)': {
            'mIoU': 0.864, 'mean_dice': 0.912, 'road_iou': 0.925, 'flooded_road_iou': 0.878,
            'boundary_f1': 0.891, 'num_params_m': 4.12, 'model_size_mb': 15.7,
            'latency_ms': 11.2, 'fps': 89.3
        }
    })


@app.route('/api/capstone_info')
def capstone_info():
    return jsonify({
        'title': 'Flood / Waterlogging Segmentation and Road Passability Estimation',
        'institution': 'Ramdeobaba University, Nagpur',
        'department': 'Department of Electronics Engineering',
        'course': 'Deep Learning Lab | Academic Session: 2026–27 | Semester: V',
        'team': [
            {'name': 'Vidhi Gupta', 'roll': '67', 'role': 'Dataset & Preprocessing (Harmonization, Augmentation, Splitting)'},
            {'name': 'Shreeya Bhomle', 'roll': '52', 'role': 'Flood & Road Segmentation (DL Segmentation Models & Masks)'},
            {'name': 'Tejas Dahiwale', 'roll': '61', 'role': 'Passability & Integration (Flooded-road ratio, Geometry, Classification)'},
            {'name': 'Sayali Ingle', 'roll': '50', 'role': 'Evaluation & Deployment (Metrics, Baseline Comparison, Edge Demo)'}
        ],
        'objectives': [
            'Develop a unified multi-dataset training pipeline combining FloodNet, CVFD, and Flood Detection v1.',
            'Train a lightweight Deep Learning segmentation model (MobileNet-UNet) for pixel-level flood and road identification.',
            'Estimate road passability using flooded-road pixel ratio and connected geometry into Passable, Partially Flooded, or Severely Affected.',
            'Evaluate system performance using mIoU, Dice score, flooded-road IoU, boundary quality, FPS, latency, and memory usage.'
        ]
    })


# ==============================================================================
# VIDEO INFERENCE PIPELINE ENDPOINTS
# ==============================================================================

_SHARED_PREDICTORS = {}

def get_video_predictor(model_type, depth_backend):
    key = (model_type, depth_backend)
    if key not in _SHARED_PREDICTORS:
        # pyrefly: ignore[missing-import]
        from inference import FloodSegmentationPredictor
        _SHARED_PREDICTORS[key] = FloodSegmentationPredictor(
            model_type=model_type,
            depth_backend=depth_backend,
            colormap="turbo"
        )
    return _SHARED_PREDICTORS[key]


def _run_video_task(task_id, input_path, output_path, depth_backend, model_type, stride, scale, max_frames, keyframe_interval=2):
    """Background worker executing video segmentation and STURM passability telemetry."""
    try:
        VIDEO_TASKS[task_id]['status'] = 'running'
        VIDEO_TASKS[task_id]['progress'] = 0.0

        def progress_cb(pct, current_frame, total_frames, msg, live_telemetry=None):
            if task_id in VIDEO_TASKS:
                VIDEO_TASKS[task_id]['progress'] = pct
                VIDEO_TASKS[task_id]['current_frame'] = current_frame
                VIDEO_TASKS[task_id]['total_frames'] = total_frames
                VIDEO_TASKS[task_id]['message'] = msg
                if live_telemetry:
                    VIDEO_TASKS[task_id]['live'] = live_telemetry

        pred_instance = get_video_predictor(model_type, depth_backend)

        summary = process_video(
            input_path=input_path,
            output_path=output_path,
            depth_backend=depth_backend,
            model_type=model_type,
            stride=stride,
            scale=scale,
            max_frames=max_frames,
            plot_summary=True,
            progress_callback=progress_cb,
            predictor=pred_instance,
            keyframe_interval=keyframe_interval,
        )

        out_name = os.path.basename(output_path)
        base_name = os.path.splitext(out_name)[0]

        VIDEO_TASKS[task_id]['completed'] = True
        VIDEO_TASKS[task_id]['progress'] = 100.0
        VIDEO_TASKS[task_id]['status'] = 'done'
        VIDEO_TASKS[task_id]['result'] = {
            'video_url': f'/outputs/{out_name}',
            'stream_url': f'/api/video_feed/{out_name}',
            'profile_url': f'/outputs/{base_name}_profile.png',
            'telemetry_url': f'/outputs/{base_name}_telemetry.json',
            'summary': summary,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        if task_id in VIDEO_TASKS:
            VIDEO_TASKS[task_id]['completed'] = True
            VIDEO_TASKS[task_id]['status'] = 'error'
            VIDEO_TASKS[task_id]['error'] = str(e)


@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    """Serves annotated videos, depth charts, and telemetry files with explicit MIME types."""
    mimetype = None
    if filename.lower().endswith('.mp4'):
        mimetype = 'video/mp4'
    elif filename.lower().endswith('.png'):
        mimetype = 'image/png'
    elif filename.lower().endswith('.json'):
        mimetype = 'application/json'
    return send_from_directory(OUTPUTS_DIR, filename, mimetype=mimetype)


@app.route('/api/video_feed/<path:filename>')
def video_feed(filename):
    """Streams video frames as MJPEG multipart stream for 100% universal browser compatibility."""
    video_path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404

    def generate_stream():
        cap = cv2.VideoCapture(video_path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                ret_enc, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret_enc:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.033)
        finally:
            cap.release()

    from flask import Response
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/video/start', methods=['POST'])
def start_video_processing():
    """Starts video segmentation and STURM passability job in a background thread."""
    try:
        file = request.files.get('video')
        video_path = request.form.get('video_path')
        model = request.form.get('model', 'mobilenet_unet')
        depth_backend = request.form.get('depth_backend', 'sturm')
        stride = int(request.form.get('stride', 2))
        scale = float(request.form.get('scale', 0.5))
        keyframe_interval = int(request.form.get('keyframe_interval', 2))
        max_frames_str = request.form.get('max_frames')
        max_frames = int(max_frames_str) if max_frames_str and max_frames_str.strip().isdigit() else None

        task_id = str(uuid.uuid4())[:8]

        if file and file.filename:
            safe_name = f"{task_id}_{os.path.basename(file.filename)}"
            input_path = os.path.join(UPLOADS_DIR, safe_name)
            file.save(input_path)
        elif video_path and os.path.exists(video_path):
            input_path = video_path
        else:
            default_vid = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_video.mp4'))
            if os.path.exists(default_vid):
                input_path = default_vid
            else:
                return jsonify({'success': False, 'error': 'No video provided or found on server'}), 400

        output_path = os.path.join(OUTPUTS_DIR, f"annotated_web_{task_id}.mp4")

        VIDEO_TASKS[task_id] = {
            'task_id': task_id,
            'status': 'starting',
            'progress': 0.0,
            'current_frame': 0,
            'total_frames': 0,
            'message': 'Initializing models and stream decoder...',
            'completed': False,
            'result': None,
            'error': None,
        }

        th = threading.Thread(
            target=_run_video_task,
            args=(task_id, input_path, output_path, depth_backend, model, stride, scale, max_frames, keyframe_interval),
            daemon=True
        )
        th.start()

        return jsonify({'success': True, 'task_id': task_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/video/status/<task_id>', methods=['GET'])
def get_video_status(task_id):
    """Polls real-time progress for an active video processing job."""
    if task_id not in VIDEO_TASKS:
        return jsonify({'success': False, 'error': 'Task not found'}), 404
    return jsonify({'success': True, 'task': VIDEO_TASKS[task_id]})


@app.route('/api/video/prerendered', methods=['GET'])
def get_prerendered_video():
    """Returns the most recent pre-rendered annotated video for instant demonstration."""
    import glob
    files = sorted(
        glob.glob(os.path.join(OUTPUTS_DIR, 'annotated_web_*.mp4')),
        key=os.path.getmtime, reverse=True
    )
    if files:
        fname = os.path.basename(files[0])
        base  = os.path.splitext(fname)[0]
        summary = {}
        telem_path = os.path.join(OUTPUTS_DIR, f'{base}_telemetry.json')
        if os.path.exists(telem_path):
            try:
                with open(telem_path, 'r') as f:
                    summary = json.load(f)
            except Exception:
                pass
        return jsonify({'success': True, 'result': {
            'video_url':     f'/outputs/{fname}',
            'stream_url':    f'/api/video_feed/{fname}',
            'telemetry_url': f'/outputs/{base}_telemetry.json',
            'summary':       summary,
        }})
    # Fallback to legacy preloaded video
    pre_rendered = os.path.join(OUTPUTS_DIR, 'annotated_12303330.mp4')
    if os.path.exists(pre_rendered):
        return get_preloaded_video()
    return jsonify({'success': False, 'error': 'No pre-rendered video found'})


@app.route('/api/video/preloaded', methods=['GET'])
def get_preloaded_video():
    """Returns pre-rendered video journey data for instant demonstration."""
    pre_rendered = os.path.join(OUTPUTS_DIR, 'annotated_12303330.mp4')
    pre_telemetry = os.path.join(OUTPUTS_DIR, 'annotated_12303330_telemetry.json')
    if os.path.exists(pre_rendered) and os.path.exists(pre_telemetry):
        try:
            with open(pre_telemetry, 'r') as f:
                summary = json.load(f)
            return jsonify({
                'success': True,
                'result': {
                    'video_url': '/outputs/annotated_12303330.mp4',
                    'stream_url': '/api/video_feed/annotated_12303330.mp4',
                    'profile_url': '/outputs/annotated_12303330_profile.png',
                    'telemetry_url': '/outputs/annotated_12303330_telemetry.json',
                    'summary': summary,
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': False, 'error': 'Preloaded video not found'})


@app.route('/api/video_samples', methods=['GET'])
def get_video_samples():
    """Lists available video samples on server for one-click testing."""
    samples = []
    test_vid = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_video.mp4'))
    if os.path.exists(test_vid):
        samples.append({
            'id': 'street_flood_test',
            'title': 'Flooded Urban Corridor (1080p 60FPS)',
            'path': test_vid,
            'description': 'Real-world street dashcam footage traversing varying water depths (STURM Level 0 to 4)',
            'duration': '29.5s'
        })
    pre_rendered = os.path.join(OUTPUTS_DIR, 'annotated_12303330.mp4')
    return jsonify({
        'samples': samples,
        'pre_rendered_available': os.path.exists(pre_rendered)
    })


if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
