<div align="center">

# 🤟 BISINDO Object Detection

**Comparative Study of Object Detection Models for Real-Time BISINDO (Indonesian Sign Language) Recognition on Edge Devices**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.52-00FFFF?logo=ultralytics&logoColor=black)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

</div>

---

## 📖 Overview

This repository contains the full training, evaluation, and analysis pipeline for my undergraduate thesis research on **real-time sign language recognition** using modern object detection architectures.

The study benchmarks **seven object detection models** trained on a custom BISINDO alphabet dataset and evaluates them on both accuracy metrics (mAP, Precision, Recall) and real-world edge-device latency using an **ESP32-CAM** deployment target.

---

## 🧠 Models Compared

| Model | Backbone | Framework |
|---|---|---|
| **YOLOv8m** | CSPDarknet | Ultralytics |
| **YOLOv9m** | GELAN | Ultralytics |
| **YOLOv10m** | CSPNet v10 | Ultralytics |
| **YOLO11m** | C3k2 | Ultralytics |
| **YOLO26m** | — | Ultralytics |
| **Faster R-CNN** | ResNet-50 + FPN | PyTorch / torchvision |
| **SSD** | VGG-16 | PyTorch / torchvision |

---

## 📁 Project Structure

```
BISINDO-Object-Detection-Training/
├── src/
│   ├── background_images/      # Augmentation background assets
│   ├── coco/                   # COCO-format annotations
│   ├── dataset_yolo/           # YOLO-format dataset (images + labels)
│   ├── dim_images/             # Dim/low-light augmented images
│   ├── esp32cam_videos/        # Raw video recordings from ESP32-CAM
│   └── extracted_frames/       # Frames extracted from ESP32-CAM videos
│
├── training_scripts/
│   ├── yolov8m.ipynb           # YOLOv8m training on Google Colab
│   ├── yolov9m.ipynb           # YOLOv9m training
│   ├── yolov10m.ipynb          # YOLOv10m training
│   ├── yolo11m.ipynb           # YOLO11m training
│   ├── yolo26m.ipynb           # YOLO26m training
│   ├── Faster RCNN.ipynb       # Faster R-CNN training (custom loop)
│   └── SSD.ipynb               # SSD training (custom loop)
│
├── evaluation_scripts/
│   ├── model_evaluation.ipynb          # Main evaluation: mAP, P, R, F1
│   ├── more_model_evaluation.ipynb     # Extended per-class analysis
│   ├── all_model_comparison.ipynb      # Cross-model comparison charts
│   ├── yolo_comparison.ipynb           # YOLO-family ablation study
│   ├── background_accuracy.ipynb       # Accuracy under background variation
│   ├── esp32cam_latency.ipynb          # Latency analysis visualization
│   ├── esp32cam_latency.py             # Real-time latency test (ESP32-CAM)
│   └── frame_extraction.py             # Frame extraction from video
│
├── backup/                     # Model checkpoints & training results backup
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Daffin-TW/BISINDO-Object-Detection-Training.git
cd BISINDO-Object-Detection-Training
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install PyTorch (with CUDA support)
Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select the appropriate command for your OS and CUDA version. Example for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Training notebooks are designed to run on **Google Colab** with GPU acceleration. The evaluation and latency scripts can be run locally.

---

## 🚀 Usage

### Training (Google Colab)
Open any notebook inside `training_scripts/` with Google Colab. Each notebook is self-contained and includes:
- Environment setup & dependency installation
- Dataset download from Google Drive
- Model training with logging
- Result export (`.zip` download)

### Local Evaluation
```bash
# Extract frames from ESP32-CAM videos
python evaluation_scripts/frame_extraction.py

# Run real-time latency test (requires connected ESP32-CAM)
python evaluation_scripts/esp32cam_latency.py
```

Open evaluation notebooks in JupyterLab or VS Code for interactive analysis.

---

## 📊 Key Evaluation Metrics

- **mAP@0.5** and **mAP@0.5:0.95** (COCO-standard)
- **Precision** and **Recall** per class
- **F1-Score**
- **Inference Latency (ms)** on ESP32-CAM stream

---

## 🔧 Hardware

| Component | Specification |
|---|---|
| Training GPU | NVIDIA (Google Colab) |
| Edge Device | ESP32-CAM (OV2640 camera module) |
| Dataset Format | YOLO & COCO |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 👤 Author

**Daffin_TW** — Undergraduate Thesis, 2026

> *"Recognizing sign language through the lens of deep learning — one gesture at a time."*
