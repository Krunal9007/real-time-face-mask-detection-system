# Real-Time Face Mask Detection System

## Project Overview
This project aims to develop a *real-time face mask detection system* that can automatically detect whether individuals are wearing face masks correctly or not.  
The system uses a combination of *Computer Vision* and *Deep Learning* techniques to identify faces in video frames and classify them as "Mask" or "No Mask".

The goal is to create a *lightweight and accurate* detection model that can run in real-time on a standard computer using a webcam or camera feed.

This repository is being developed as part of the *IU International University of Applied Sciences* course:
> *Project: Computer Science (CSEMCSPCSP01)* – Phase 2 (Development/Reflection Phase)

---

## Objectives
- Build a deep learning model that can detect whether a person is wearing a mask or not.  
- Achieve reliable detection speed and accuracy for real-time use.  
- Implement the model using *Python, **OpenCV, and **TensorFlow/Keras*.  
- Evaluate performance based on accuracy, precision, recall, and F1-score.  
- Prepare documentation and report as per IU university portfolio guidelines.

---

## Technical Approach

### 1. Data Collection
- Using the [Face Mask Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection).  
- Dataset includes labeled images for *Mask* and *No Mask* classes.  
- Data will be preprocessed and augmented to increase model robustness.

### 2. Preprocessing
- Resize all images to 128x128 pixels.  
- Normalize pixel values (0–1 range).  
- Split dataset: 80% Training, 10% Validation, 10% Testing.  

### 3. Model Development
- *Model Type:* Convolutional Neural Network (CNN)  
- *Frameworks:* TensorFlow, Keras  
- *Activation Functions:* ReLU for hidden layers, Softmax for output layer  
- *Loss Function:* Binary Crossentropy  
- *Optimizer:* Adam  

### 4. Training
- Model will be trained on GPU (if available).  
- Training progress (loss/accuracy) will be tracked and visualized using Matplotlib.  
- Model weights will be saved in the models/ folder.  

### 5. Real-Time Detection
- Integration with *OpenCV* for real-time video feed processing.  
- Use Haar cascades or DNN-based face detectors to locate faces in each frame.  
- For each detected face, classify mask status using the trained CNN model.

---

## Testing & Evaluation
Testing will be conducted on:
- *Static images* (for model accuracy evaluation)
- *Real-time webcam feed* (for performance in dynamic environments)

Metrics to be used:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Frame processing rate (FPS)

---

## Project Structure
```bash
Mask-Detection-System/
│
├── data/ # Datasets (or dataset links)
├── notebooks/ # Jupyter notebooks for experiments
├── src/ # Python source code
│ ├── preprocess.py
│ ├── train_model.py
│ ├── detect_mask.py
│ └── utils.py
├── models/ # Trained models (.h5 files)
├── results/ # Plots, metrics, or sample outputs
├── README.md # Project documentation
├── requirements.txt # Required Python libraries
└── report_draft.docx # Phase 2 draft report
```bash
