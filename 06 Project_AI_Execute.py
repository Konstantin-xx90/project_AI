import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ----------------- CONSTANTS -----------------

# Specify the path to your best trained model
# IMPORTANT: Use the final stable model name!
MODEL_PATH = "final_stable_emotion_model.keras"

# Define the expected input image size (must match the model's training)
IMG_SIZE = (48, 48)

# Define the emotion classes (order must match the model's output order)
EMOTION_CLASSES = ["Angry", "Fear", "Happy", "None"]

# ----------------- LOAD MODEL -----------------

try:
    # Load the trained model from disk
    model = load_model(MODEL_PATH)
    print(f"Model successfully loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading the model: {e}")
    print("Please ensure the file 'final_stable_emotion_model.keras' exists in this directory.")

def predict_emotion(image_path):
    """
    Predicts the emotion from a grayscale face image.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        tuple: (str: Predicted Label, np.ndarray: Prediction probabilities)
    """
    # Load the input image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 1. Image Preprocessing
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0  # Normalization
    # Reshape: (1, 48, 48, 1) - Batch size 1, Height, Width, 1 Channel (Grayscale)
    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    # 2. Prediction
    # verbose=0 suppresses the Keras output like '1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step'
    prediction = model.predict(img, verbose=0)[0]
    emotion_index = np.argmax(prediction)

    # 3. Result
    detected_emotion = EMOTION_CLASSES[emotion_index]

    return detected_emotion, prediction


def display_prediction(image_path, detected_emotion, probabilities):
    """
    Displays the image along with the predicted label and probabilities.
    """
    # Re-load the image to display in color (or its original format)
    img_display = cv2.imread(image_path)

    if img_display is None:
        print(f"Visualization image not found: {image_path}")
        return

    # OpenCV reads as BGR, Matplotlib expects RGB
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    # Formatting the probabilities for display
    prob_text = [f"{EMOTION_CLASSES[i]}: {p * 100:.1f}%" for i, p in enumerate(probabilities)]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(img_display)

    # Title with the result and all probabilities
    title = f"Predicted Emotion: {detected_emotion}\n"
    title += "\n".join(prob_text)

    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.show()


# ----------------- TEST EXECUTION -----------------

# NOTE: REPLACE THIS WITH THE PATH TO YOUR TEST IMAGE!
TEST_IMAGE_PATH = "/Users/konstantinhanemann/PycharmProjects/Project_AI/test_image.jpg"

try:
    predicted_label, probs = predict_emotion(TEST_IMAGE_PATH)

    print("-" * 40)
    print(f"Result for '{TEST_IMAGE_PATH}':")
    print(f"-> Detected Emotion: {predicted_label}")
    print("-" * 40)

    display_prediction(TEST_IMAGE_PATH, predicted_label, probs)

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

