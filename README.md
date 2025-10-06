# Project: Artificial Intelligence - Python Project 

## Table of Contents
1. Purpose
2. Requirements
3. Usage
4. Code Structure for further Usage

## Purpose
The main goal of the task is the detection of the emotional state of a person that looks at a company’s
advertisement. Therefore, an algorithm that accurately classifies the three emotional expressions
joy, anger, or fear will be developed. The implementation of the project will be done with the help of
Python in PyCharm. The final product can detect one of the three emotional states or none when
receiving an unlabelled picture of a face.

This application was created due as a part of a university project (Data Science).

### 2. Requirements
- Python 3.8 or higher
- Use the Requirements.txt file, there are listed all required libraries

### 3. Usage
1. Install Python (at least Python 3.8) on your computer with a respective Development Environment
(Find the latest version [here](https://www.python.org/downloads/)).
2. Download the file "final_stable_emotion_model.keras"
3. Load the code from ```Project_AI_Execute.py``` into your development IDE and run for classifying a new picture that is secured as picture on your computer.
   Please adjust the path in the code correctly to the picture and to the trained program under step 2.
4. The result will be shown as an output.

### 4. Code Structure for further Usage
#### For further Usage, please also find the structure of the dataset and the trained programm with its parameters

1. Download the picture dataset from [here](https://www.kaggle.com/datasets/msambare/fer2013)
2. Save the dataset in a folder and secure the data in the folder structure as followed
3. ![image](https://github.com/user-attachments/assets/3b11c283-be2d-4b22-ab76-04717c1a4f63)  
   That means, the images with different tags must be copied in ```none```, for Training and Testing.
4. Load the code from ```project_AI.py``` to your development IDE and run to train the algorithm. Please adjust the path in the code correctly.
5. Then, the model can be trained accordingly and paramenters can be adjusted as wished.
