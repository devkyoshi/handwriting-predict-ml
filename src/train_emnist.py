import numpy as np
from scipy.io import loadmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load the .mat file
mat = loadmat("emnist/emnist-balanced.mat")

# Extract training and testing data
training_images = mat["dataset"]["train"][0,0]["images"][0,0]
training_labels = mat["dataset"]["train"][0,0]["labels"][0,0]
test_images = mat["dataset"]["test"][0,0]["images"][0,0]
test_labels = mat["dataset"]["test"][0,0]["labels"][0,0]

# Reshape and normalize
X_train = training_images.reshape(-1, 28, 28).astype("float32") / 255.0
X_test = test_images.reshape(-1, 28, 28).astype("float32") / 255.0

# Rotate -90 degrees and flip horizontally to fix orientation
X_train = np.transpose(X_train, (0, 2, 1))[:, ::-1, :]
X_test = np.transpose(X_test, (0, 2, 1))[:, ::-1, :]

# Add channel dimension
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# One-hot encode the labels
num_classes = len(np.unique(training_labels))
y_train = to_categorical(training_labels, num_classes)
y_test = to_categorical(test_labels, num_classes)

# Define CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

# Save the trained model
model.save("model/emnist_model.h5")

print("✅ Model trained and saved as model/emnist_model.h5")
