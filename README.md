# Flood / Waterlogging Segmentation and Road Passability Estimation

**Ramdeobaba University, Nagpur**  
*Department of Electronics Engineering*  
**Course:** Deep Learning Lab | **Academic Session:** 2026–27 | **Semester:** V  

---

## 📌 Project Overview & Problem Statement

Flooding can submerge roads and make them unsafe or completely impassable, disrupting transportation and delaying emergency responses. Traditional manual identification from CCTV or drone imagery is time-consuming and prone to human error under changing weather and visibility conditions.

This Capstone Project develops an automated **Deep Learning Multi-Modal Passability Pipeline** combining:
1. **Semantic Road & Water Segmentation** (MobileNet-UNet Lite & Baseline U-Net) for pixel-accurate 2D road corridor and floodwater coverage.
2. **STURM-FloodDepth Physical Flood Level Classifier** ([Zenodo: 14833532](https://zenodo.org/records/14833532)) to classify flood depth into 5 calibrated physical levels (Level 0 to 4) using vehicle landmarks instead of ungrounded relative depth models.
3. **Connected Road Geometry Passability Engine** to classify road passability as **Passable**, **Partially Flooded**, or **Severely Affected** in real-time.

---

## 👥 Team Information & Work Distribution

| Sr. No. | Team Member | Roll No. | Major Responsibility |
|---|---|---|---|
| **1** | **Vidhi Gupta** | **67** | **Dataset & Preprocessing:** Data collection, label harmonization, augmentation pipeline, 70/15/15 split. |
| **2** | **Shreeya Bhomle** | **52** | **Flood & Road Segmentation:** Architecture design, PyTorch model training (MobileNet-UNet & U-Net), loss formulation. |
| **3** | **Tejas Dahiwale** | **61** | **Passability & Integration:** Flooded-road ratio algorithm, connected geometry breakdown, passability classification. |
| **4** | **Sayali Ingle** | **50** | **Evaluation & Deployment:** Quantitative benchmarking (mIoU, Dice, FPS, Latency), failure mode analysis, Web App GUI demo. |

---

## 🚗 STURM-FloodDepth: Physical Depth Grounding

We integrated **STURM-FloodDepth** ([Zenodo DOI: 10.5281/zenodo.14833532](https://zenodo.org/records/14833532)). STURM uses submerged vehicle features (tires, wheels, chassis, windows) as physical reference objects to directly categorize flood severity into **5 calibrated physical depth levels**:

| Level | Physical Depth Range | Vehicle Landmark Reference | Passability Verdict | Driving Advisory |
|---|---|---|---|---|
| **Level 0** | **0 cm** (Dry) | Tires and wheels fully dry; no water contact | 🟢 **Passable** | Road is dry and clear. Normal driving caution. |
| **Level 1** | **1 – 15 cm** | Water below wheel axle / less than half tire submerged | 🟢 **Passable** | Minor shallow water (<15 cm). Safe for all vehicles. |
| **Level 2** | **15 – 30 cm** | Water covering wheel hub / half to full tire | 🟡 **Partially Flooded** | Caution: Clearance required. SUVs / Trucks only. Sedans avoid. |
| **Level 3** | **30 – 60 cm** | Water reaching chassis, exhaust, door sills, or window base | 🔴 **Severely Affected** | UNSAFE: Engine hydrolock & stall hazard. Impassable! |
| **Level 4** | **> 60 cm** (60–120+ cm) | Water reaching/exceeding window line; submerged vehicle | 🔴 **Severely Affected** | EMERGENCY: Deep inundation (>60 cm). Vehicle submerged. Road closed! |

### Advantages of STURM:
- **Physically Grounded:** Eliminates arbitrary scaling constants.
- **Ultra-Lightweight & Fast:** MobileNetV3-Small backbone (~1.5M parameters, 6.2 MB), running at **>100 FPS** on GPU and **<20 ms** on CPU (compared to 25M–335M parameters for ViT).
- **Dual-Stream Multi-Modal Fusion:** 2D spatial road coverage ratio from MobileNet-UNet + 3D vertical depth level from STURM-FloodDepth.

---

## 🏷️ Multi-Dataset Label Harmonization Taxonomy

Harmonized label schema across FloodNet (10 classes), CVFD (3 classes), and Flood Detection v1:

| Class Index | Class Name | Description | BGR Color Code |
|---|---|---|---|
| **0** | Background | Non-road terrain, buildings, trees, sky, vehicles, assets | [40, 44, 52] (Dark Grey) |
| **1** | Road | Dry, passable paved road surface | [50, 205, 50] (Emerald Green) |
| **2** | Water / Floodwater | Submerged road, standing waterlogging, flood pools | [0, 140, 255] (Amber / Cyan) |

---

## 🏗️ System Architecture & Methodology

```text
+------------------------------------+     +-----------------------------------+
|     Input Street Flood Image       |     |     Multi-Dataset Ingestion       |
+------------------------------------+     +-----------------------------------+
          |                      |
          v                      v
+--------------------+   +-----------------------------------+
|  MobileNet-UNet    |   | STURM-FloodDepth (MobileNetV3)    |
|  2D Segmentation   |   | 5-Class Depth (Level 0 to 4)      |
+--------------------+   +-----------------------------------+
          |                      |
          |  Mask (Road, Water)  |  Level & Calibrated Depth (cm)
          v                      v
+------------------------------------------------------------+
|        Multi-Modal Passability Fusion & HUD Engine         |
|  Status: Passable / Partially Flooded / Severely Affected  |
+------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------+
|            Interactive Flask Dashboard & Telemetry         |
|     Visual Overlay | Bathymetric Heatmap | Vehicle HUD     |
+------------------------------------------------------------+
```

---

## 📁 Repository Structure

```text
dl_project/
├── data/
│   ├── raw/                              # Segmentation benchmark data (images & masks)
│   ├── sturm/                            # STURM-FloodDepth dataset (Level0..Level4)
│   └── test_video.mp4                    # Urban street flood video sample (1080p)
├── weights/
│   ├── best_mobilenet_unet.pth           # Lightweight segmentation model (1.2 MB)
│   └── best_flood_level_model.pth        # STURM 5-class flood depth model (6.2 MB)
├── outputs/
│   ├── annotated_12303330.mp4            # Pre-rendered annotated demonstration video
│   ├── annotated_12303330_profile.png    # Longitudinal depth profile chart
│   ├── annotated_12303330_telemetry.json # Frame-by-frame telemetry
│   ├── evaluation_report.json            # Quantitative benchmark report
│   ├── history_mobilenet_unet.json       # Epoch-wise training loss, mIoU & Dice history
│   └── training_history_mobilenet_unet.png # Training loss & metric curves
├── src/
│   ├── models/
│   │   ├── unet.py                       # Baseline U-Net Architecture
│   │   ├── mobilenet_unet.py             # Lightweight MobileNet-UNet Architecture
│   │   ├── sturm_classifier.py           # STURM 5-Class MobileNetV3-Small Classifier
│   │   └── loss.py                       # Combined Cross-Entropy + Dice Loss
│   ├── utils/
│   │   ├── dataset.py                    # DataLoader, Splitting, Augmentation
│   │   ├── harmonize.py                  # Multi-dataset label mapping
│   │   ├── metrics.py                    # mIoU, Dice, Boundary F1, Latency, FPS, Params
│   │   ├── passability.py                # Dual-Stream Passability Engine & HUD visual overlay
│   │   ├── sturm_estimator.py            # STURM-FloodDepth Depth Estimator & HUD Card Generator
│   │   ├── depth_estimator.py            # Bathymetric depth colorization & heatmap utilities
│   │   └── download_sturm.py             # Automated STURM dataset downloader (Zenodo 14833532)
│   ├── train.py                          # Segmentation model training script
│   ├── train_sturm.py                    # STURM flood depth level training script
│   ├── evaluate.py                       # Quantitative benchmark evaluation script
│   ├── inference.py                      # Single image CLI inference runner
│   └── inference_video.py                # Video stream segmentation & HUD processing engine
├── web_app/
│   ├── app.py                            # Flask REST API server (multi-backend depth)
│   ├── templates/index.html              # Modern dark-mode Capstone Dashboard GUI
│   └── static/                           # Glassmorphic CSS styling & Chart.js logic
├── run_demo.py                           # One-click demo web app launch script
├── test_pipeline.py                      # Comprehensive verification test suite
├── train_flood_level_colab.ipynb         # Interactive Google Colab notebook for STURM training
└── requirements.txt                      # Dependencies
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Complete Pipeline
```bash
python test_pipeline.py
```

### 3. Run Inference with STURM Flood Depth (Single Image)
```bash
python src/inference.py --input data/raw/images/sample_0001.jpg --depth-backend sturm
```

### 4. Process Urban Flood Video Stream
```bash
# Annotate an urban flood video with MobileNet-UNet & STURM HUD passability telemetry
python src/inference_video.py --input data/test_video.mp4 --depth-backend sturm --output outputs/annotated_run.mp4
```

### 5. Train STURM-FloodDepth Model (Zenodo: 14833532)
```bash
# Automatically download dataset from Zenodo (or place folders in data/sturm)
python src/utils/download_sturm.py --output-dir data/sturm

# Train or fine-tune MobileNetV3-Small on 5 flood levels
python src/train_sturm.py --data-dir data/sturm --epochs 15 --batch-size 32 --lr 1e-3
```

### 6. Launch Interactive Web App Dashboard
```bash
python run_demo.py
```

Open **http://127.0.0.1:5000** in your browser to:
- Test live drag-and-drop image uploads.
- Process urban flood video footage with real-time progress tracking.
- Switch between **STURM-FloodDepth (Vehicle Reference 0-4)** and **Morphological Bathymetry**.
- View real-time passability classification, vehicle clearance advisories, and depth heatmaps!
#   G r o u p _ 7 _ V - A _  
 