import os
import cv2
import numpy as np
import mahotas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load sandboil images and compute features
def load_and_extract_features(directory):
    images = []
    hu_moments_features = []
    haralick_features = []
    for filename in os.listdir(directory):
        image = cv2.imread(os.path.join(directory, filename))
        if image is not None:
            images.append(image)
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Compute Hu Moments
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log-transform Hu Moments to make them scale-invariant
            hu_moments_features.append(-np.sign(hu_moments) * np.log10(np.abs(hu_moments)))
            # Compute Haralick texture features using mahotas
            haralick = mahotas.features.haralick(gray).mean(axis=0)
            haralick_features.append(haralick)
    return images, hu_moments_features, haralick_features

# Load and extract features from positive (sandboil) and negative (non-sandboil) images
positive_images, positive_hu_moments, positive_haralick = load_and_extract_features('positive_images/images')
negative_images, negative_hu_moments, negative_haralick = load_and_extract_features('negative_images')

# Combine features into a single feature vector for each image
X_hu = np.vstack((positive_hu_moments + negative_hu_moments))
X_haralick = np.vstack((positive_haralick + negative_haralick))
X = np.hstack((X_hu, X_haralick))
y = np.concatenate([np.ones(len(positive_hu_moments)), np.zeros(len(negative_hu_moments))])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize feature vectors
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_normalized.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_normalized, y_train, epochs=100, batch_size=20, validation_split=0.5)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_normalized, y_test)
print(f'Test Accuracy: {test_acc}')

# Save the trained model
model.save('sandboil_model_modified.keras')
