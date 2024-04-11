import cv2
import numpy as np
import os

# Load positive and negative images
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        image = cv2.imread(os.path.join(directory, filename))
        if image is not None:
            images.append(image)
    return images

positive_images = load_images('positive_images')
negative_images = load_images('negative_images')

# Preprocess images (resize, convert to grayscale)
target_size = (100, 100)
positive_images = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), target_size) for img in positive_images]
negative_images = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), target_size) for img in negative_images]

# Extract features (e.g., HOG features)
# For simplicity, let's use the pixel intensity values as features
positive_features = np.array([img.flatten().astype(np.float32) for img in positive_images])
negative_features = np.array([img.flatten().astype(np.float32) for img in negative_images])

# Create labels (1 for positive, 0 for negative)
positive_labels = np.ones(len(positive_features), dtype=np.int32)
negative_labels = np.zeros(len(negative_features), dtype=np.int32)

# Merge positive and negative features
features = np.concatenate((positive_features, negative_features))
labels = np.concatenate((positive_labels, negative_labels))

# Create the SVM classifier
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

# Train the classifier
svm.train(features, cv2.ml.ROW_SAMPLE, labels)

# Save the trained classifier to a file
svm.save('sand_boil_classifier.xml')
