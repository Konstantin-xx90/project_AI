# Project: Artificial Intelligence - Python Project 

## Table of Contents
1. Purpose
2. Installation
3. Usage
4. Tests
5. Contribution

## Purpose
The main goal of the task is the detection of the emotional state of a person that looks at a company’s
advertisement. Therefore, an algorithm that accurately classifies the three emotional expressions
joy, anger, or fear will be developed. The implementation of the project will be done with the help of
Python in PyCharm. The final product can detect one of the three emotional states or none when
receiving an unlabelled picture of a face.

This application was created due as a part of a university project (Data Science).

##Requirements & Installation

###Requirements
- Python 3.8 or higher
- Tensorflow Library for Detecting Images
- Numpy, cv2, and Matplotlib

###Installation
1. Install Python (at least Python 3.8) on your computer
(Find the latest version [here](https://www.python.org/downloads/)).
2. Install the following modules within Python:
   ```ImageDataGenerator``` from ```tensorflow.keras.preprocessing.image```
   ```Sequential```from ```tensorflow.keras.models```
   ```Conv2D, MaxPooling2D, Flatten, Dense, Dropout```from ```tensorflow.keras.layers```
   ```Adam``` from ```tensorflow.keras.optimizers```
   ```Matplotlib```
   ```Numpy```
   ```cv2```

**The Program is ready to run now**

## Usage

1. Download all the documents in folder project_AI
2. Download the picture dataset from [here](https://www.kaggle.com/datasets/msambare/fer2013)
3. Save the dataset in a folder and secure the data in the folder structure as shown
4. ![image](https://github.com/user-attachments/assets/3b11c283-be2d-4b22-ab76-04717c1a4f63) means, the images with different tags must be copied in ```none```, for Training and Testing.
5. Start the document ```project_AI.py```.
6. Type in your Terminal ```python main.py```.
7. Follow the instruction on your screen.
8. When the program starts, a list of all currently created habits is displayed.
9. In the main menu, habits can be created, deleted, changed and incremented.
10. In the analysis tool, current streaks and a list of created habits can be viewed.
11. The program is not case-sensitive.
12. To check a habit, it has to be incremented in the respective time interval (daily / weekly).
13. If you forgot to submit a check, it can also be done retrospectively.

##Contribution
This was my first Python Project. Any improvements or remarks from your side can help me. \
Feel free to help me to improve the application or reporting any detected bug. Thank you :)
