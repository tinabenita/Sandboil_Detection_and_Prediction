# Portfolio Case Study: Sandboil Detection and Prediction Using Computer Vision

## 1. Project Overview

This project explores the use of computer vision and machine learning to detect and classify sandboils from images. A sandboil is a visible surface defect that can indicate underlying erosion or instability in pavement, embankment, or soil structures. The goal of the project was to build an automated inspection system that could identify these defects from visual data, compare multiple detection approaches, and create a practical foundation for future predictive monitoring systems.

The repository brings together four complementary computer-vision approaches:

- A Viola-Jones cascade detector for rapid object detection
- An SVM classifier trained on Histogram of Oriented Gradients (HOG) features
- A CNN classifier trained on handcrafted image features such as Hu moments and Haralick texture features
- A YOLOv8-based classification pipeline for modern deep-learning inference

This makes the project valuable not only as a detection system but also as a comparative study of classical computer-vision techniques versus modern deep learning.

---

## 2. Problem Statement

Manual inspection of surface defects is time-consuming, inconsistent, and often dependent on human judgment. In infrastructure or geotechnical monitoring contexts, defects such as sandboils can be subtle and difficult to identify early. The project addresses a real-world need for an automated visual inspection system that can:

- detect the presence of sandboils in images,
- reduce the dependence on manual review,
- support faster decision-making,
- create a scalable framework for future deployment in surveillance, road inspection, or civil infrastructure monitoring.

The problem is inherently visual, making computer vision an appropriate solution.

---

## 3. Project Goals

The project was designed with several goals in mind:

1. Build a reliable image-based sandboil detection pipeline.
2. Compare different machine-learning and deep-learning methods.
3. Explore feature engineering techniques such as Hu moments and Haralick texture analysis.
4. Demonstrate an end-to-end workflow from dataset preparation to model inference.
5. Create a portfolio-ready example of applied computer vision and AI.

---

## 4. Why the Project Matters

This project is important because it demonstrates how AI can be applied to real-world infrastructure and environmental monitoring tasks. Instead of relying purely on manual inspection, the system can act as a first-pass automated detector. That makes it a strong example of responsible AI in civil and environmental applications, where early detection can reduce risk and improve monitoring efficiency.

It also showcases a broad skill set:

- image preprocessing
- feature extraction
- classical machine learning
- deep learning
- model training and evaluation
- computer-vision deployment workflows

---

## 5. Dataset and Data Preparation

The project uses a curated image dataset organized into positive and negative samples:

- Positive images: images believed to contain sandboils
- Negative images: images without sandboils
- Test datasets: held-out images used for evaluation

The dataset is distributed across the project folders:

- [cnn](cnn)
- [svm](svm)
- [violajones](violajones)
- [yolov8](yolov8)

Each approach uses the same general idea of supervised learning, but the input representation differs.

### Data Preparation Practices

The repository shows that the project followed a standard computer-vision workflow:

- loading images from directories,
- resizing images to fixed dimensions,
- converting images to grayscale where necessary,
- extracting meaningful features,
- combining features into training vectors,
- splitting data into training and testing sets,
- training classifiers,
- evaluating predictions on unseen images.

This demonstrates good machine-learning discipline and makes the project suitable for portfolio presentation.

---

## 6. Technical Approach

### A. Viola-Jones Cascade Detector

The Viola-Jones approach is one of the earliest and most influential object-detection methods. In this project, it was used to detect visible sandboil-like regions using a pre-trained cascade classifier trained on Haar-like features.

#### What was implemented

The script in [violajones/sand.py](violajones/sand.py) loads a cascade classifier and applies it to test images. It uses:

- grayscale conversion,
- scale-factor tuning,
- minimum-neighbor tuning,
- minimum-size constraints,
- bounding box drawing for detections.

#### Why it was used

Viola-Jones is lightweight and historically important for real-time face and object detection. Although it is not the most powerful modern solution for complex texture-based defects, it is a useful baseline method and provides insight into the problem of object localization.

#### Strengths

- fast and simple,
- useful for quick detection baselines,
- helpful for understanding classical computer-vision pipelines.

#### Limitations

- sensitive to scale, lighting, and image quality,
- less suitable for highly variable or subtle defects than modern deep-learning methods.

---

### B. SVM with HOG Features

The SVM pipeline uses Histogram of Oriented Gradients (HOG), a classic feature descriptor that captures shape and edge information. The project trains a support vector machine on HOG features extracted from resized images.

#### What was implemented

The script [svm/createDataseHOG.py](svm/createDataseHOG.py) performs the following steps:

- loads positive and negative images,
- resizes them to a fixed target size,
- computes HOG features,
- creates labels,
- trains an SVM classifier,
- saves the model as [svm/sand_boil_classifier_with_hog.xml](svm/sand_boil_classifier_with_hog.xml).

The inference script [svm/svmwithhog.py](svm/svmwithhog.py) then loads the trained SVM model and predicts whether a given test image contains a sandboil.

#### Why it was used

HOG features are strong for capturing structural patterns, and SVMs are effective classifiers when the data is relatively small and well-structured. This approach is a strong classical baseline.

#### Strengths

- interpretable and lightweight,
- effective when dataset size is moderate,
- good for demonstrating feature-based machine learning.

#### Limitations

- performance depends heavily on preprocessing and feature representation,
- may struggle with complex textures and noisy real-world images.

---

### C. CNN with Hu Moments and Haralick Features

This is one of the most interesting parts of the project. Instead of feeding raw image pixels directly into the network, the model uses handcrafted descriptors that capture geometry and texture.

#### Feature Engineering

The training script [cnn/cnn_train_withHu.py](cnn/cnn_train_withHu.py) extracts:

- Hu moments: shape-based descriptors that describe image geometry and contour information,
- Haralick features: texture statistics derived from the gray-level co-occurrence matrix.

These features are then concatenated into a single feature vector and passed into a compact neural network.

#### Model Architecture

The CNN used in this script is a simple feed-forward neural network with:

- an input layer matching the feature vector size,
- a dense hidden layer with ReLU activation,
- dropout regularization,
- a sigmoid output layer for binary classification.

This is a lightweight approach that combines classic feature engineering with modern neural-network learning.

#### Why it was used

This approach is valuable because it demonstrates that machine-learning performance does not always require raw pixel-based training. It shows the ability to blend domain knowledge with neural networks.

#### Strengths

- interpretable feature set,
- effective for compact datasets,
- demonstrates hybrid modeling.

#### Limitations

- feature extraction may require domain tuning,
- the model architecture is intentionally simple and may not generalize to larger or more diverse datasets without improvement.

---

### D. YOLOv8 Classification Pipeline

The YOLOv8 portion represents the most modern and scalable part of the project. The implementation in [yolov8/predict.py](yolov8/predict.py) loads a trained classification model and predicts the class of a supplied image.

#### Training Configuration

The training configuration file [yolov8/runs/classify/train/args.yaml](yolov8/runs/classify/train/args.yaml) shows that the model used:

- YOLOv8n classification architecture,
- pretrained weights,
- 5 training epochs,
- image size 224,
- batch size 16,
- classification task mode.

#### Training Results

The training results file [yolov8/runs/classify/train/results.csv](yolov8/runs/classify/train/results.csv) shows strong performance:

- top-1 accuracy reached 98.735% at epoch 5,
- training loss decreased from 0.32078 to 0.06727,
- validation loss also improved over training.

These results are strong evidence that the YOLOv8 classification model learned the visual pattern effectively during the training run.

#### Why it matters

YOLOv8 is widely used in modern object detection and classification tasks. Including it in the project shows that the author is comfortable working with production-grade deep-learning pipelines, not just traditional machine-learning baselines.

---

## 7. End-to-End Workflow

The project follows a clear workflow that can be presented as a professional machine-learning pipeline:

1. Data Collection
   - gather positive and negative images,
   - organize them into categories.

2. Preprocessing
   - resize and normalize images,
   - convert to grayscale when appropriate,
   - prepare consistent input formats.

3. Feature Extraction
   - use handcrafted descriptors for classical models,
   - use image-based input for the YOLOv8 model.

4. Model Training
   - fit classical classifiers and neural networks,
   - optimize based on the selected representation.

5. Evaluation
   - rely on standard metrics such as accuracy, F1, precision, recall, ROC-AUC, and confusion matrix.

6. Inference and Visualization
   - apply trained models to test images,
   - display predictions and bounding boxes.

This workflow is exactly what employers and reviewers look for in applied AI projects.

---

## 8. What the Repository Demonstrates

This project is not just a single model implementation. It demonstrates the ability to:

- compare multiple modeling strategies,
- work with both classical and modern computer-vision techniques,
- engineer features manually when useful,
- integrate deep-learning models into a practical workflow,
- structure a project around a clear problem domain.

That breadth makes the repository especially strong for a portfolio.

---

## 9. Tools and Technologies Used

The project uses a practical and relevant toolkit for applied computer vision:

- Python
- OpenCV
- NumPy
- scikit-learn
- TensorFlow/Keras
- Mahotas
- Ultralytics YOLOv8
- Matplotlib and Tkinter for visualization and basic display

These are widely used technologies in industry and research settings, which strengthens the project’s relevance.

---

## 10. Challenges and Lessons Learned

Every applied AI project has limitations, and this one is no exception. The repository shows that the project faced several realistic challenges:

- small or limited datasets can make generalization difficult,
- image quality and lighting variations affect performance,
- classical methods can be sensitive to scale and appearance changes,
- deep-learning results depend heavily on training data quality,
- hard-coded file paths and manual preprocessing reduce portability.

These challenges are valuable because they demonstrate maturity. A strong portfolio project does not need to be perfect; it should show that the builder understands the real-world constraints of AI development.

---

## 11. Strengths of the Project

This project stands out for several reasons:

- It solves a meaningful visual inspection problem.
- It compares multiple methods rather than relying on a single approach.
- It shows a progression from traditional machine learning to deep learning.
- It includes both training and inference scripts.
- It uses standard evaluation practices.
- It is structured clearly enough to be understood by recruiters and technical reviewers.

---

## 12. Areas for Improvement

If this project were to be expanded for a more advanced portfolio version, the following improvements would make it even stronger:

- collect a larger and more diverse dataset,
- annotate images more precisely and consistently,
- add data augmentation,
- train and compare models with proper validation splits and cross-validation,
- report final metrics more systematically,
- convert the workflow into a reusable pipeline,
- deploy the model as a web app or desktop application,
- integrate explainability and visualization of detections.

These are excellent next steps for turning the project into a more production-ready solution.

---

## 13. Why This Project Is Strong for a Portfolio

This project is a strong portfolio piece because it demonstrates real-world AI application, technical breadth, and practical problem-solving. It shows that the author can:

- understand a domain problem,
- build a complete computer-vision pipeline,
- work with multiple modeling strategies,
- evaluate models critically,
- and present a professional, research-oriented project.

It is especially valuable because it does not only show “model training”; it shows a full workflow from raw images to decision support.

---

## 14. Short Portfolio Summary

This project develops an automated sandboil detection system using computer vision and machine learning. It compares classical techniques such as Viola-Jones, HOG + SVM, and handcrafted feature-based CNNs with a modern YOLOv8 classification pipeline. The work demonstrates strong applied AI skills in image preprocessing, feature extraction, model training, and evaluation, while also highlighting the practical challenges of building reliable vision systems for real-world inspection tasks.

---

## 15. Suggested Portfolio Description

A concise version you can use on GitHub or LinkedIn:

“Built an end-to-end computer vision project for sandboil detection using OpenCV, scikit-learn, TensorFlow, and YOLOv8. The project compares classical methods such as Viola-Jones and HOG + SVM with a feature-based CNN and a modern YOLOv8 classification pipeline. It demonstrates practical experience in image preprocessing, feature engineering, model training, evaluation, and deployment-oriented inference workflows.”
