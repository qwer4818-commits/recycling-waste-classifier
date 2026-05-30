# Recycling Waste Classifier

**EF1003 — Term Project 1 · KENTECH**

A CNN-based model that automatically classifies six categories of recyclable waste
(plastic, paper, glass, metal, styrofoam, trash) from a single image.

🌐 **Project page:** [Live demo & writeup](https://qwer4818-commits.github.io/minsuu_Kang/project1.html)

---

## Authors
- **Minsu Kang** — Student ID 20230460 · Team Lead
- **Jeyoon Song** — Class of 2026

---

## Project Overview

We wanted to make something useful for everyday life. Korea's actual recycling rate
is lower than expected, so we built an AI that helps sort waste at the source.

### What we did
1. Collected 1,057 raw images (self-taken + Kaggle + web)
2. Cleaned and preprocessed them (224×224, balanced to 300 per class)
3. Trained a CNN from scratch (3 conv blocks → FC → 6 classes)
4. Evaluated with confusion matrix, per-class accuracy, and error analysis
5. Built a real-time webcam demo

---

## Files

```
source/
├── preprocess.py   # Clean data and prepare images for training
├── train.py        # Train the CNN model
├── evaluate.py     # Test the model and generate result plots
└── demo.py         # Real-time webcam classification
```

---

## Large Files (Google Drive)

Due to GitHub file size limits, the trained model and dataset are hosted externally.

📦 [Download model + dataset](https://drive.google.com/drive/folders/1zwtSAvs60y2OmTc43DCsvnyh1UQ4RDgF?usp=drive_link)

Contents:
- `models/best_model.pth` (196 MB) — trained CNN weights (58% val accuracy)
- `data/` — raw, processed, and split image data (4,613 files)

---

## Setup

```bash
pip3 install torch torchvision opencv-python pillow imagehash matplotlib scikit-learn numpy
```

Place downloaded files like this:

```
project/
├── source/
├── models/best_model.pth
├── data/
│   ├── raw/
│   ├── processed/
│   └── split/
└── README.md
```

---

## Usage

```bash
python3 source/preprocess.py   # optional — data already preprocessed
python3 source/train.py        # optional — model already trained (~5 min on GPU)
python3 source/evaluate.py     # requires best_model.pth
python3 source/demo.py         # requires best_model.pth + webcam
```

---

## Model

A clean from-scratch CNN — no pretrained weights.

| Block | Layers |
|-------|--------|
| Conv block 1 | Conv2d(32) → ReLU → MaxPool |
| Conv block 2 | Conv2d(64) → ReLU → MaxPool |
| Conv block 3 | Conv2d(128) → ReLU → MaxPool |
| Head | FC(512) → Dropout(0.5) → FC(6) |

**Training:** 50 epochs · batch 32 · Adam · lr 0.001 · CrossEntropy

---

## Results

**Final accuracy: 58%**

| Class | Accuracy | Notes |
|-------|----------|-------|
| Styrofoam | 95.6% | Distinct white color & texture |
| Metal | ~70% | Metallic shine helps |
| Trash | ~60% | High visual variability |
| Glass | ~50% | Confused with plastic |
| Paper | ~40% | Overlaps with cardboard |
| Plastic | ~40% | Looks like glass |

The main error source: transparent items (plastic ↔ glass) are hard to separate
without pretrained features.

---

## Future Work
- Transfer learning (ResNet, EfficientNet) for stronger baseline
- 1,000+ diverse images per class with varied lighting
- Mobile app deployment for real-time inference
- Multimodal input (RGB + weight / material sensors)

---

## References
1. Chungnam National University (2023). Study on Korea's actual recycling rates.
2. Lee, J. (2019). Global recycling rates and policy analysis.
3. Song, H. (2024). OECD plastic waste statistics.
4. Kaggle — Garbage Classification Dataset
5. PyTorch — Deep Learning with PyTorch

---

## Acknowledgement

This project was completed as part of **EF1003 Data Literacy Foundation** at KENTECH.
We thank Prof. Ukcheol Shin for his lectures and feedback throughout the project.
