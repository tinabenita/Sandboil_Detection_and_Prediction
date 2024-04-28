import cv2
import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

def detect_sand_boils(image):
    classifier_path = 'sand_boil_classifier_with_hog.xml'

    # Load the classifier
    classifier = cv2.ml.SVM_load(classifier_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to match the training data size (64x128)
    resized_image = cv2.resize(gray, (64, 128))

    # Compute HOG features
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hog_features = hog.compute(resized_image).flatten().astype(np.float32)

    # Reshape the feature vector to match the SVM model's input
    feature_vector = hog_features.reshape(1, -1)

    # Predict using the SVM model
    _, prediction = classifier.predict(feature_vector)

    return prediction[0, 0]
# Extract the prediction value

# Get a list of all the image files in the directory.
image_files = os.listdir('test_datasets')

# Load ground truth labels
ground_truth_labels = np.loadtxt('ground_truth_labels.txt', dtype=int)

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate over the list of image files.
for image_file, ground_truth_label in zip(image_files, ground_truth_labels):
    # Load the image.
    image = cv2.imread(os.path.join('test_datasets', image_file))

    # Check that the image is not None
    if image is None:
        print(f'Image {image_file} could not be loaded')
        continue

    try:
        # Perform sand boil detection
        prediction = detect_sand_boils(image)

        # Update true and predicted labels
        true_labels.append(ground_truth_label)
        predicted_labels.append(prediction)

        # If prediction is positive (1), draw bounding box or perform further processing
        if prediction == 1:
            # Process the detected sand boil (e.g., draw bounding box around it).
            # For demonstration, let's just print a message.
            print(f'Sand boil detected in image: {image_file}')
            cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
        else:
            continue

        # Display the image with the detected sand boil.
        cv2.imshow('Sand Boil Detection', image)
        # Position the window to the center of the screen
        screen_width = tk.Tk().winfo_screenwidth()
        screen_height = tk.Tk().winfo_screenheight()
        window_width, window_height = image.shape[1], image.shape[0]
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        cv2.moveWindow('Sand Boil Detection', x, y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred while processing image {image_file}: {e}")
        continue

# Calculate accuracy, F1 score, and confusion matrix
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Extract values from confusion matrix
TN, FP, FN, TP = conf_matrix.ravel()

# Compute recall and precision
recall = TP / (TP + FN)
precision = TP / (TP + FP)

# Calculate False Positive Rate (FPR)
FPR = FP / (FP + TN)

# Compute AUC
fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
roc_auc = auc(fpr, tpr)

# Print recall, precision, AUC, and FPR
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'AUC: {roc_auc}')
print(f'False Positive Rate: {FPR}')


# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
