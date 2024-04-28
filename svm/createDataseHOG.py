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

positive_images = load_images('positive_images\images')
negative_images = load_images('negative_images')

# Define target size for resizing
target_size = (64, 128)

# Preprocess images (resize)
positive_images_resized = [cv2.resize(img, target_size) for img in positive_images]
negative_images_resized = [cv2.resize(img, target_size) for img in negative_images]

# Extract HOG features
hog = cv2.HOGDescriptor()
# Adjust HOG parameters as needed
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # Example of using default HOG parameters
# Compute HOG features for positive images
positive_features = np.array([hog.compute(img).flatten() for img in positive_images_resized])
# Compute HOG features for negative images
negative_features = np.array([hog.compute(img).flatten() for img in negative_images_resized])

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
svm.save('sand_boil_classifier_with_hog.xml')
