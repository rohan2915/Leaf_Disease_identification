🌽 Leaf Disease Identification System
An end-to-end machine learning system for detecting and managing crop diseases in maize (corn), combining state-of-the-art computer vision with an epidemiological risk assessment engine.

Overview
This project contains two complementary systems:
SystemFileTaskArchitectureMulticlass Disease Classifiermaize_disease_classifier.pyClassifies 4 maize diseases from leaf imagesEfficientNetB3 + FPN + AttentionGLS Detection & Risk Enginegls_detection/gls_detection.pyBinary GLS detection + spray recommendationMobileNetV2 + Weather Risk Model

System 1 — Multiclass Maize Disease Classifier
Detects four classes from leaf images:

✅ Healthy
🟡 Gray Leaf Spot (GLS)
🟠 Northern Leaf Blight
🔴 Common Rust

Architecture

Backbone: EfficientNetB3 pretrained on ImageNet, top 30 layers fine-tuned
Feature Pyramid Network (FPN): Multi-scale feature extraction with lateral connections and bilinear upsampling for capturing disease patterns at multiple resolutions
Attention Mechanism: Spatial attention gating to focus on disease-relevant leaf regions
Classification Head: Residual dense blocks (1024 → 512) with BatchNorm and LeakyReLU

Training Techniques

Mixup Augmentation (α=0.2) — interpolates training samples to improve generalisation
Random Erasing (p=0.3) — randomly masks image regions to prevent overfitting
Cosine Annealing with Warmup — 5-epoch warmup then cosine LR decay over 30 epochs
Label Smoothing (0.1) — softens targets to reduce overconfident predictions
Class Weight Balancing — handles dataset imbalance across disease classes
Standard Augmentation: rotation, width/height shift, shear, zoom, channel shift, brightness variation

Evaluation
Metrics tracked: Accuracy, Precision, Recall, Confusion Matrix (raw + normalised)

System 2 — Gray Leaf Spot Detection & Spray Recommendation Engine
A two-stage pipeline: image-based disease detection followed by weather-driven risk assessment to recommend fungicide application timing.
Stage 1 — Image Classifier

Backbone: MobileNetV2 pretrained on ImageNet (lightweight, deployment-friendly)
Task: Binary classification — Healthy vs. Gray Leaf Spot
Training: Standard augmentation + Early Stopping + ModelCheckpoint

Stage 2 — Epidemiological Risk Model (GLSRiskAssessor)
Computes daily GLS disease risk from hourly weather data using agronomic thresholds:
ParameterLow RiskOptimal (High Risk)Temperature15–32°C20–28°CRelative Humidity70–86%86–96%Leaf Wetness< 5 hrs> 5 hrs
Risk levels (High / Medium / Low / No Risk) accumulate over rolling windows:

7-day sum ≥ 9 → Spray recommended
10-day sum ≥ 8 → Spray recommended
14-day sum ≥ 10 → Spray recommended

CLI Interface
Run gls_detection.py to:

Train a new model from your own image dataset
Predict disease from a single leaf image
Assess risk using weather data (manual input or defaults)


Tech Stack
Python · TensorFlow / Keras · EfficientNetB3 · MobileNetV2
scikit-learn · OpenCV · Hugging Face · Matplotlib · Seaborn · Pandas · NumPy
Google Colab (training environment)

Project Structure
Leaf_Disease_identification/
├── maize_disease_classifier.py   # Multiclass EfficientNetB3 classifier
├── gls_detection/
│   ├── gls_detection.py          # Binary GLS classifier + risk engine
│   └── Untitled-1.py             # Experimentation notebook
├── docs/                         # Additional documentation
├── index.html                    # Web interface
├── script.js                     # Frontend logic
└── styles.css                    # Styling

Getting Started
bash# Clone the repo
git clone https://github.com/rohan2915/Leaf_Disease_identification.git
cd Leaf_Disease_identification

# Install dependencies
pip install tensorflow==2.12.0 opencv-python scikit-learn seaborn matplotlib pandas numpy

# Run the multiclass classifier (designed for Google Colab)
# Open maize_disease_classifier.py in Colab and run cells sequentially

# Run the GLS detection system
python gls_detection/gls_detection.py
