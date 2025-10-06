"""
Emotion Detection Model Training and Testing Script

This script trains a Convolutional Neural Network (CNN) to recognize emotions 
from grayscale facial images. It uses TensorFlow and Keras for deep learning.
"""


# 1.1 Import necessary deep learning and plotting libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import seaborn as sns
import numpy as np

# Set up of further Markdown possibilities (only required for JupyterLab)
def printmd(string):
    display(Markdown(string))

# 1.2 Define the paths to the datasets
# NOTE: Please ensure these paths are correct on your system!
train_dir = "/Users/konstantinhanemann/PycharmProjects/Project_AI/archive/train"
test_dir = "/Users/konstantinhanemann/PycharmProjects/Project_AI/archive/test"

# 1.3 Define global training parameters
img_size = (48, 48)
batch_size = 64

printmd("**Display of the Class indices**")
# Training Generator: Scaling and reserving 20% for validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=10,        # Small Rotation
    width_shift_range=0.0,   # Horizontal Switch
    height_shift_range=0.0,  # Vertical Switch
    shear_range=0.0,          # Shear
    zoom_range=0.0,          # Zoom
    horizontal_flip=True,     # Horizontal Mirroring
    fill_mode='nearest'
)

# Test Generator: Scaling only (no augmentation for test data!)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training Set Generator: Loads images and maps them to classes
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale", # Grayscale images
    class_mode="categorical",
    subset="training"
)

# Validation Set Generator
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# Test Set Generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)


aggressive_class_weights = {
    0: 2.0,  # Angry: High weight
    1: 2.5,  # Fear: Highest weight due to critical failure (0.05 Recall)
    2: 1.0,  # Happy: Moderate weight
    3: 0.8   # None: Ultra-low weight (was 0.5 in earlier suggestions)
}

# Calculate class weights to equalize the weights
true_labels = train_generator.classes
unique_classes = np.unique(true_labels)

# Calculate the weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=true_labels
)

class_weights = dict(enumerate(weights))
class_counts = np.bincount(true_labels)

printmd("<br>**Calculated Class-weights**")
print("Examples of each class:", class_counts)
print("Class-Weights (Index: Weight):", class_weights)

# Display the class indices (mapping from name to number)
num_classes = len(train_generator.class_indices)

printmd("<br>**Class Information**")
print("Number of Classes:", num_classes)
print("Class Mapping:", train_generator.class_indices)

# Get the dynamic number of classes
num_classes = len(train_generator.class_indices)

# Create the sequential model architecture
model = Sequential([
    # First Conv Block: 32 Filters
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    # Second Conv Block: 64 Filters
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.6), # Dropout for regularization

    # Third Conv Block: 128 Filters
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.6),

    # Classification Part
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.7),
    Dense(num_classes, activation="softmax") # Output Layer: Softmax for classification
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001), # Explicit Adam optimizer
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Display a summary of the model
printmd("**Summary of the CNN Architecture**")
model.summary()

# Start the training process

# Define the Early Stopping Callback
# monitor='val_loss': Monitor the validation loss (should be minimized)
# patience=10: Wait 10 epochs with no improvement before stopping training
# restore_best_weights=True: Restores the weights from the epoch with the best 'val_loss'
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# Define the Model Checkpoint Callback for saving the best model
# monitor='val_accuracy': Monitor the validation accuracy (should be maximized)
# save_best_only=True: Only saves the model if the metric is better than before
model_checkpoint = ModelCheckpoint(
    filepath='best_emotion_model_checkpoint.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Create the list of callbacks
callbacks_list = [early_stopping, model_checkpoint]

print(f"\nStarting training over epochs...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=150,
    callbacks=callbacks_list,

    class_weight=aggressive_class_weights
)
print("Training complete.")

# Evaluate the model on the test dataset
printmd("**Model Evaluation**")
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot of accuracy during training
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy", color='blue')
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color='orange')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Note on interpretation: A large gap between the lines suggests overfitting.

EMOTION_CLASSES = ["Angry", "Fear", "Happy", "None"]

# 1. Get Predictions on the validation dataset
validation_steps = val_generator.n // val_generator.batch_size + 1

# Raw predictions (Probabilities)
Y_pred = model.predict(val_generator, steps=validation_steps)
# Convert probabilities to class labels (index of the highest probability)
y_pred_classes = np.argmax(Y_pred, axis=1)

# Retrieve the true labels of the validation dataset
y_true = val_generator.classes

# 2. Create the Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

# 3. Visualize the Confusion Matrix
plt.figure(figsize=(10, 8))
# Use EMOTION_CLASSES from your script here
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix for Emotion Recognition')
plt.show()

# 4. Output the detailed classification report (next section)
print(classification_report(y_true, y_pred_classes, target_names=EMOTION_CLASSES))

# Save the trained model in Keras format
final_model = load_model("best_emotion_model_checkpoint.keras")
final_model.save("final_stable_emotion_model.keras")
