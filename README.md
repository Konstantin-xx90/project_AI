# Project: Artificial Intelligence - Python Project 

## Table of Contents
1. Purpose
2. Installation
3. Usage

## Purpose
The main goal of the task is the detection of the emotional state of a person that looks at a company’s
advertisement. Therefore, an algorithm that accurately classifies the three emotional expressions
joy, anger, or fear will be developed. The implementation of the project will be done with the help of
Python in PyCharm. The final product can detect one of the three emotional states or none when
receiving an unlabelled picture of a face.

This application was created due as a part of a university project (Data Science).

## Requirements & Installation

### Requirements
- Python 3.8 or higher
- Tensorflow Library for Detecting Images
- Numpy, cv2, and Matplotlib

### Usage
1. Install Python (at least Python 3.8) on your computer
(Find the latest version [here](https://www.python.org/downloads/)).
2. Install the following modules within Python:
   ```ImageDataGenerator``` from ```tensorflow.keras.preprocessing.image```/
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
3. Save the dataset in a folder and secure the data in the folder structure as followed
4. ![image](https://github.com/user-attachments/assets/3b11c283-be2d-4b22-ab76-04717c1a4f63) means, the images with different tags must be copied in ```none```, for Training and Testing.
5. Start the document ```project_AI.py``` and run to train the algorithm.
7. Run ```Project_AI_Execute.py```for classifying a new picture.
8. The result will be shown in your console.
