# Sandboil Detection and Prediction

## Project Overview

This project develops an automated sandboil detection system using computer vision and machine learning. A sandboil is a visible surface defect that can indicate underlying erosion in infrastructure. 

## Four Detection Approaches

The repository brings together four complementary computer-vision methods:

- **Viola-Jones Cascade Detector** - Rapid object detection using classical cascade classifiers
- **SVM with HOG Features** - Support vector machine trained on Histogram of Oriented Gradients
- **CNN with Handcrafted Features** - Neural network using Hu moments and Haralick texture descriptors
- **YOLOv8 Classification** - Modern deep-learning model achieving 98.7% accuracy

## Problem Statement

Manual inspection of surface defects is time-consuming, inconsistent, and dependent on human judgment. This project addresses the need to:

- Detect sandboils automatically in images
- Reduce manual review workload
- Enable faster decision-making
- Create a scalable framework for infrastructure and geotechnical monitoring

## Project Structure

```
cnn/                           CNN with Hu moments and Haralick features
├── cnn_train_withHu.py       Training script
├── cnn_test_withhu.py        Testing and inference
├── sandboil_model_with_hu_haralick_features.keras
└── datasets/                 Training and test images

svm/                           SVM with HOG features
├── createDataseHOG.py        Training script
├── svmwithhog.py             Inference script
├── sand_boil_classifier_with_hog.xml
└── datasets/                 Training and test images

yolov8/                        YOLOv8 deep learning
├── Train.ipynb               Training notebook
├── predict.py                Inference script
└── runs/                      Training results and configs

violajones/                    Viola-Jones cascade detector
├── sand.py                   Detection script
└── datasets/                 Test images

portfolio_case_study.md        Detailed project documentation
```

## Key Technologies

- **Language:** Python
- **Computer Vision:** OpenCV
- **Machine Learning:** scikit-learn, TensorFlow/Keras
- **Feature Extraction:** Mahotas, NumPy
- **Deep Learning:** Ultralytics YOLOv8
- **Visualization:** Matplotlib, Tkinter

## Dataset

The project uses curated image datasets organized into:
- Positive samples: images containing sandboils
- Negative samples: images without sandboils
- Test sets: held-out evaluation images

Datasets are distributed across each model's directory for independent evaluation.

## End-to-End Workflow

1. **Data Collection** - Organize positive and negative sample images
2. **Preprocessing** - Resize, normalize, and convert to appropriate formats
3. **Feature Extraction** - Compute handcrafted features or prepare raw pixels
4. **Model Training** - Train classifiers using the selected representation
5. **Evaluation** - Assess performance using accuracy, F1, precision, recall, and ROC-AUC
6. **Inference** - Apply trained models to new test images

## Model Performance

### YOLOv8 Results (Best Performance)
- Top-1 Accuracy: 98.735% at epoch 5
- Training Loss: 0.32078 → 0.06727
- Configuration: YOLOv8n, 224x224 images, 16 batch size, 5 epochs

### Approach Comparison

| Method | Type | Strengths | Limitations |
|--------|------|-----------|------------|
| Viola-Jones | Classical | Fast, interpretable, good baseline | Sensitive to scale/lighting |
| HOG + SVM | Classical ML | Lightweight, effective for moderate data | Depends on preprocessing |
| CNN + Handcrafted Features | Hybrid | Interpretable, works on small datasets | May not generalize broadly |
| YOLOv8 | Deep Learning | State-of-the-art performance | Requires more data, less interpretable |

## Getting Started

### Prerequisites
```bash
pip install opencv-python numpy scikit-learn tensorflow keras ultralytics mahotas matplotlib
```
