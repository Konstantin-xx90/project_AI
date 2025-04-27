"""
Predict Emotion from a Face Image using a Pretrained CNN Model.

Author: Konstantin (or your name)
Date: 2025-04-27
"""

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model from disk
model = load_model("emotion_detection_model.keras")

# Define the expected input image size (must match the model's training)
img_size = (48, 48)

# Define the emotion classes (order must match the model's output order)
EMOTION_CLASSES = ["Angry", "Fear", "Happy", "None"]

def predict_emotion(image_path):
    """
    Predicts the emotion from a grayscale face image.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: Predicted emotion label.

    Displays:
        The input image with the predicted emotion as the title.
    """
    # Load the input image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Resize and normalize the image
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0  # Ensure float division
    img = img.reshape(1, img_size[0], img_size[1], 1)  # Reshape for model input

    # Predict emotion using the model
    prediction = model.predict(img, verbose=0)
    emotion_index = np.argmax(prediction)

    # Retrieve the corresponding emotion label
    detected_emotion = EMOTION_CLASSES[emotion_index]

    # Display the original image with prediction
    img_display = cv2.imread(image_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    plt.imshow(img_display, cmap="gray")
    plt.title(f"Predicted Emotion: {detected_emotion}")
    plt.axis("off")
    plt.show()

    return detected_emotion

# Test with an unknown picture
emotion = predict_emotion("test_image.jpg")
print(f"Detected Emotion: {emotion}")

