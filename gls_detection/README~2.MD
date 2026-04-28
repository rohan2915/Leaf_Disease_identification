# Gray Leaf Spot (GLS) Detection & Management System

A binary leaf disease classifier combined with a weather-driven epidemiological risk assessment engine to help farmers decide when to apply fungicide treatment.

---

## What It Does

This module does two distinct things:

1. **Classifies leaf images** as Healthy or Gray Leaf Spot infected using MobileNetV2 transfer learning
2. **Assesses disease spread risk** using hourly weather data and recommends fungicide spray timing

---

## Image Classifier

- **Model:** MobileNetV2 (ImageNet pretrained) + GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(64) → Softmax
- **Classes:** Healthy · Gray Leaf Spot
- **Training:** Data augmentation (rotation, shift, shear, zoom, horizontal flip), Early Stopping, ModelCheckpoint
- **Evaluation:** Accuracy, Precision, Recall, Confusion Matrix, Classification Report

---

## Epidemiological Risk Engine (`GLSRiskAssessor`)

Models GLS disease spread using established agronomic thresholds for temperature, relative humidity, and leaf wetness duration.

### Daily Risk Calculation

| Condition | Temperature | Humidity | Result |
|---|---|---|---|
| High Risk | 20–28°C | 86–96% RH | High (score: 2) |
| Medium Risk | 15–32°C | 70–86% RH | Medium (score: 1) |
| Low Risk | 20–28°C | < 70% RH | Low (score: 0.5) |
| — | Any other | Any other | No Risk (score: 0) |

Leaf wetness (threshold: > 5 hrs wet) further adjusts risk:
- High risk + insufficient wetness → downgraded to Medium
- Medium risk + insufficient wetness → downgraded to Low

### Spray Recommendation Logic

Daily risk scores accumulate over rolling windows:

| Window | Threshold | Decision |
|---|---|---|
| 7 days | Sum ≥ 9 | Spray recommended |
| 10 days | Sum ≥ 8 | Spray recommended |
| 14 days | Sum ≥ 10 | Spray recommended |

---

## Usage

```bash
python gls_detection.py
```

Three modes available:

```
1. Prepare dataset and train a new model
2. Use existing model for prediction on a single image
3. Run risk assessment only (no model required)
```

### Mode 1 — Train
Point to folders of healthy and GLS leaf images. The script splits them 70/15/15 (train/val/test), trains MobileNetV2, evaluates, and saves the model.

### Mode 2 — Predict
Load a saved `.h5` model and pass any leaf image to get a prediction with confidence score and treatment recommendation.

### Mode 3 — Risk Assessment
Input hourly temperature, humidity, and leaf wetness data (or use defaults). Get a daily risk level, visualise rolling risk history, and receive a spray recommendation.

---

## Tech Stack

```
Python · TensorFlow / Keras · MobileNetV2 · scikit-learn
Matplotlib · Pandas · NumPy · shutil · pathlib
```

---

## Files

| File | Description |
|---|---|
| `gls_detection.py` | Main system: classifier + risk engine + CLI |
| `Untitled-1.py` | Experimental notebook / prototyping |
