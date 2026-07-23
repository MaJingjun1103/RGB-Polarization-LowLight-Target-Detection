# RGB-Polarization Fusion Based Low-Light Target Enhancement and Saliency Detection System
## Project Overview
This project develops a multi-modal vision system for **low-light target enhancement and saliency detection**, integrating RGB images, full-spectrum intensity images, and polarization information.

The system aims to improve target visibility under challenging conditions including:
- low illumination
- low contrast
- complex backgrounds
- camouflage-like scenes

The complete pipeline consists of:

RGB-W fusion → Color restoration → Low-light enhancement → Polarization-guided saliency detection

---

# Key Features

## 1. RGB and Full-spectrum Image Fusion

A RGB-W fusion framework is developed to combine:

- RGB color information
- full-spectrum grayscale intensity information

A Laplacian pyramid based fusion strategy is adopted to transfer:

- structural details
- brightness information
- local contrast

from intensity images into RGB images.

---

## 2. Low-light Color Restoration

To correct illumination-induced color distortion, the system integrates:

- Black level compensation
- White balance correction
- Color correction matrix (CCM)
- Nonlinear color mapping network

A lightweight LUNA-based enhancement network is introduced for nonlinear color restoration.

The network adopts:

- Lightweight U-Net generator
- Depthwise separable convolution
- Attention mechanism
- Two-stage training strategy

including:

Stage 1:
Supervised reconstruction learning

Stage 2:
GAN-based detail refinement

---

## 3. RGB Color Saliency Modeling

RGB images are transformed into LAB color space.

Color contrast features are extracted to construct RGB saliency maps, improving discrimination between:

- target
- background

---

## 4. Polarization-guided Target Enhancement

Polarization information is introduced to improve detection capability in:

- low contrast scenes
- complex illumination environments

The polarization branch utilizes:

- Degree of Linear Polarization (DoLP)
- polarization contrast
- edge information

---

## 5. Adaptive Multi-feature Fusion

A dynamic fusion strategy is proposed by considering:

- RGB contrast
- polarization response
- noise level
- edge strength

Adaptive weights are calculated to balance RGB and polarization contributions.

Scharr operator is employed for edge enhancement.

---

## 6. Post-processing and Target Refinement

A target refinement module is designed using:

- connected component analysis
- region constraints
- sparsity constraints

to suppress noise and improve target integrity.

---

# Project Structure

RGB-Polarization-LowLight-System/

│
├── main.py # Main pipeline entry
├── train.py # LUNA enhancement network training
├── test.py # Single image inference
├── detect.py # Saliency detection module
├── fusion.py # RGB-polarization feature fusion
├── enhance_image.py # Low-light enhancement interface
├── evaluate.py # Evaluation tools
│
├── data/
│ ├── dataset.py # Dataset definition
│ ├── dataloader.py # Data loading
│ └── augmentation.py # Data augmentation
│
├── dataL/
│ ├── dataset.py
│ ├── dataloader.py
│ └── augmentation.py
│
├── models/
│ ├── generator.py # LUNA enhancement generator
│ ├── discriminator.py # PatchGAN discriminator
│ └── losses.py # Training losses
│
├── baselines/
│ ├── crd.py
│ ├── lrx.py
│ ├── lrasr.py
│ ├── lsmad.py
│ ├── pta.py
│ └── ssiifd.py
│
├── utils/
│ ├── metrics.py
│ ├── color.py
│ ├── edge.py
│ └── normalize.py
│
├── experiments/
│ └── ablation and comparison experiments
│
├── deploy/
│ └── ONNX/TensorRT deployment tools
│
├── requirements.txt
└── README.md