# Amateur Tennis CV System

This project provides an end-to-end computer vision pipeline for analyzing amateur tennis matches.  
It combines modern object detection, tracking, and homography calibration to generate match insights such as player tracking, ball tracking, hit detection, and per-player statistics.

---

## Features

### Player Detection & Tracking
- Uses **YOLOv8** for detection and **ByteTrack** for persistent player IDs.

### Ball Tracking
- **GridTrackNet** model predicts tennis ball positions using heatmaps.

### Minicourt Overlay
- Court perspective is calibrated via **homography**. Player and ball positions are projected onto a minimap.

### Hit Detection
- Events are detected when the ball approaches a player and changes direction.

### Output Video
- Annotated video includes:
  - Player bounding boxes
  - Ball trajectory
  - Minimap overlay
  - Per-player hit counts

---

## Requirements

- **Anaconda or Miniconda**
- **Python 3.10** (used for development)
- *(Optional)* NVIDIA GPU with CUDA for faster inference
- **ffmpeg** installed and available in PATH (required for video processing)

---

## Installation

Clone the repository and create the environment from the provided `environment.yml` file:

```bash
git clone https://github.com/AmineBennani7/amateur-tennis-cv-system.git
cd amateur-tennis-cv-system

# Create the conda environment
conda env create -f environment.yml

# Activate it
conda activate tennis-cv
```

⚡ **Note on GPU**  
The provided `environment.yml` installs the **CPU version** of PyTorch by default.  
If you have an NVIDIA GPU, please follow the official PyTorch installation instructions to install the CUDA-enabled build.

Example (for CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation
Run the following quick tests to make sure the environment is working:

```bash
# Test YOLOv8 detection on a sample image
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')(source='https://ultralytics.com/images/bus.jpg')"

# Test TensorFlow / GridTrackNet
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
```

---

## Project Structure

```
amateur-tennis-cv-system/
├─ main_tracking.py                # Main pipeline: detection, tracking, minimap, hits
├─ videos_cut.py                   # Cut or slice long input videos
├─ extract_frames.py               # Extract frames for dataset preparation
├─ hit_detect.ipynb                # Jupyter notebook for testing hit detection logic
├─ main_yolo_inference.ipynb       # Test YOLO models with visualization
│
├─ constants/                      # Global constants (court dimensions, etc.)
├─ main_utils/                     # Utility scripts
│   ├─ calibration.py              # Homography calibration (manual court corners)
│   ├─ detection.py                # Detection and tracking integration
│   ├─ visualisation.py            # Overlays, drawing, minimap
│
├─ model/                          # YOLO and GridTrackNet trained weights
├─ data/                           # Input videos
├─ calibration/                    # Saved homography matrix (after calibration)
├─ metrics/                        # Evaluation scripts and results
├─ trackers/                       # ByteTrack and StrongSORT tracker configs
├─ utils/                          # Helper functions
│
├─ runs/                           # Output videos and logs
├─ environment.yml                 # Conda environment definition
└─ README.md                       # Project documentation
```

---

## Usage

### 1. Prepare Input Videos
- Place raw match videos in the `data/` directory.
- Optionally, use `videos_cut.py` to split long recordings into shorter clips.

### 2. Court Calibration
Run `calibration.py` (inside `main_utils/`) to manually select the four court corners.  
A homography matrix is saved in the `calibration/` folder for later use.

### 3. Run the Main Pipeline
Execute the main tracking script:

```bash
python main_tracking.py --input data/match.mp4 --output runs/output.mp4
```

The script performs player detection & tracking, ball tracking, hit detection, and generates a video with annotations.

---

## Output
- Processed videos with overlays are saved inside the `runs/` directory.

---

## Evaluation
- Scripts in the `metrics/` folder allow evaluation of detection and tracking performance using metrics such as **precision, recall, MOTA, and MOTP**.
- Ground-truth labels can be provided in **YOLO** or **MOTChallenge** format for evaluation.
