"""
Emotion Detection Model Training and Testing Script

This script trains a Convolutional Neural Network (CNN) to recognize emotions 
from grayscale facial images. It uses TensorFlow and Keras for deep learning.
"""


# Import required libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define dataset paths
train_dir = "/Users/konstantinhanemann/PycharmProjects/Project_AI/archive/train"
test_dir = "/Users/konstantinhanemann/PycharmProjects/Project_AI/archive/test"

# Define target image size and batch size
img_size = (48, 48)
batch_size = 64

# Define data generators with rescaling and validation split for training
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Define data generator for testing (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# Load test images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

# Print class labels mapping
print("Class Labels:", train_generator.class_indices)

# Get the number of classes dynamically
num_classes = len(train_generator.class_indices)

# Create CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")  # Output layer with softmax for emotion categories
])

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=28,
)

# Evaluate model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot accuracy graph for training and validation
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Save the trained model in Keras format
model.save("emotion_detection_model.keras")
