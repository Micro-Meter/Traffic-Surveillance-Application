import os
import numpy as np
import tensorflow as tf
import cv2

LABELS = ['Car', 'Bike', 'Truck']
DATASET_PATH = 'vehicle_dataset'
IMG_WIDTH = 100
IMG_HEIGHT = 120

EPOCHS = 10
BATCH_SIZE = 32

#load data
def load_data():
    images = []
    labels = []

    for label_index, label_name in enumerate(LABELS):
        folder_path = os.path.join(DATASET_PATH, label_name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img_resized)
            labels.append(label_index)

    return np.array(images), np.array(labels)

#load and preprocess data
images_X, labels_y = load_data()
images_X = images_X.astype('float32') / 255.0  # Normalize to [0,1]
images_X = np.expand_dims(images_X, axis=-1)   # Add channel dimension: (N, 100, 100, 1)

# One-hot encode labels
labels_y = tf.keras.utils.to_categorical(labels_y, num_classes=len(LABELS))

#manually shuffle and split (80% train, 20% validation)
indices = np.arange(len(images_X))
np.random.seed(42)
np.random.shuffle(indices)
images_X = images_X[indices]
labels_y = labels_y[indices]
split = int(0.8 * len(images_X))
X_train, X_val = images_X[:split], images_X[split:]
y_train, y_val = labels_y[:split], labels_y[split:]

#CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(LABELS), activation='softmax')
])

#compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(X_val, y_val))


os.makedirs('model', exist_ok=True)
model.save('model/vehicle_cnn_model.h5')
print("Model successfully saved")
