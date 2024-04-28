import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import mahotas
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import tkinter as tk
import matplotlib.pyplot as plt

# Load testing images
def load_test_data(directory):
    images = []
    for filename in os.listdir(directory):
        image = cv2.imread(os.path.join(directory, filename))
        if image is not None:
            images.append(image)
    return images

# Load testing data
test_images = load_test_data('test_datasets')

# Load the pre-trained model
model = tf.keras.models.load_model('sandboil_model_with_hu_haralick_features.keras')

# Function to extract Hu Moments features from images
def extract_hu_moments_features(images):
    hu_moments_features = []
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute Hu Moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log-transform Hu Moments to make them scale-invariant
        hu_moments_features.append(-np.sign(hu_moments) * np.log10(np.abs(hu_moments)))
    return np.array(hu_moments_features)

# Function to extract Haralick texture features from images
def extract_haralick_features(images):
    haralick_features = []
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute Haralick texture features
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        haralick_features.append(haralick)
    return np.array(haralick_features)

# Extract features from test images
hu_moments_features = extract_hu_moments_features(test_images)
haralick_features = extract_haralick_features(test_images)

# Check the dimensions of each feature
print("Hu Moments Features shape:", hu_moments_features.shape)
print("Haralick Features shape:", haralick_features.shape)

# Combine features into a single feature vector
test_features_concatenated = np.hstack((hu_moments_features, haralick_features))

# Reshape the concatenated features to match the expected input shape of the model
test_features_reshaped = test_features_concatenated.reshape((len(test_features_concatenated), -1))

# Normalize feature vectors
scaler = StandardScaler()
test_features_normalized = scaler.fit_transform(test_features_reshaped)

# Make predictions on testing data
predictions = model.predict(test_features_normalized)

# Define the threshold
threshold = 0.9

# Output predictions based on threshold
true_labels = []

# Load ground truth labels from a separate file
ground_truth_file = 'ground_truth_labels.txt'
with open(ground_truth_file, 'r') as file:
    lines = file.readlines()
    ground_truth = np.array([int(line.strip()) for line in lines])

for i, (prediction, image) in enumerate(zip(predictions, test_images)):
    if prediction > threshold:
        print(f"Image {i+1}: Sandboil detected (Probability: {prediction[0]:.4f})")
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
    else:
        print(f"Image {i+1}: No sandboil detected (Probability: {prediction[0]:.4f})")
       

    cv2.imshow(f"Detected Image {i+1}", image)
    
    # Position the window to the center of the screen
    screen_width = tk.Tk().winfo_screenwidth()
    screen_height = tk.Tk().winfo_screenheight()
    window_width, window_height = image.shape[1], image.shape[0]
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    cv2.moveWindow(f"Detected Image {i+1}", x, y)
    
    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ground truth labels
    true_labels.append(ground_truth[i])

# Convert probabilities to binary predictions based on threshold
binary_predictions = (predictions > threshold).astype(int)

# Calculate accuracy, F1 score, and confusion matrix
accuracy = accuracy_score(true_labels, binary_predictions)
f1 = f1_score(true_labels, binary_predictions)
conf_matrix = confusion_matrix(true_labels, binary_predictions)

# Compute precision
precision = precision_score(true_labels, binary_predictions)

# Compute recall
recall = recall_score(true_labels, binary_predictions)

# Compute AUC
auc = roc_auc_score(true_labels, predictions)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predictions)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'AUC: {auc}')
print(f'Confusion Matrix:\n{conf_matrix}')
