# Real-Time Face Mask Detection System

## Project Overview
This project implements a fully working *real-time face mask detection system* capable of identifying whether individuals are wearing face masks or not.  
The complete pipeline—dataset preparation, model training, and real-time detection—was successfully developed and tested.

The system uses a combination of *Computer Vision* and *Deep Learning* techniques to detect faces in live video frames and classify them into two categories:
- **Mask**
- **No Mask**

The model has been trained and deployed locally, and the real-time detection runs smoothly on a standard computer using a webcam.  
This project was completed as part of the:

> **Project: Computer Science (CSEMCSPCSP01)** – Phase 3 (Final Submission)

---

## Objectives (Phase 3 Status)
- ✔ Build a CNN model to detect mask vs. no mask  
- ✔ Achieve reliable real-time detection performance  
- ✔ Implement using *Python, OpenCV, TensorFlow/Keras*  
- ✔ Test detection on live webcam feed  
- ✔ Document all development steps for the IU portfolio  

While no formal accuracy metrics (precision/recall/F1) were calculated, the model performs consistently well during informal testing.

---

## Technical Approach

### 1. Dataset
- Dataset used: **Face Mask Detection Dataset (Kaggle)**
- Three original classes:
  - *with_mask*
  - *without_mask*
  - *mask_weared_incorrect*  
- For this project, only “with_mask” and “without_mask” were used to form a binary classification dataset.
- The dataset is **not included** in this repository due to Kaggle licensing.

### 2. Preprocessing
- All images were resized to **128×128 pixels**
- Pixel values were normalized to the 0–1 range
- Dataset split:
  - **80% Training**
  - **20% Validation**  
- Data augmentation applied (rotation, zoom, and brightness shifts) to reduce overfitting

### 3. Model Development
- A **Convolutional Neural Network (CNN)** was designed for the binary classification task
- Implemented using **TensorFlow / Keras**
- Key components:
  - Activation functions: **ReLU** (hidden layers), **Sigmoid** (output)
  - Loss function: **Binary Crossentropy**
  - Optimizer: **Adam**
- The model was trained locally on CPU and the final weights were saved in the `/models` directory

### 4. Real-Time Detection
- Implemented using **OpenCV**
- Haar cascade classifier used for face detection
- Each detected face region is:
  1. Extracted  
  2. Preprocessed  
  3. Passed to the trained CNN model  
- Output label (“Mask” / “No Mask”) is displayed on the webcam feed in real time

---

## Testing & Evaluation

### Informal Testing
Since the project focuses on practical implementation rather than benchmarking, evaluation was performed informally through real-time trials.

Tests were done under different:
- Lighting conditions  
- Angles and distances  
- Mask types  
- Backgrounds  

### Observations
- The face detection reacts quickly and updates smoothly  
- The mask classifier provides stable predictions  
- Performance is noticeably dependent on lighting  
- Haar cascade may miss faces at extreme angles  
- Despite these limitations, the system functions reliably for typical webcam usage

---

## Project Structure
```bash
Mask-Detection-System/
│
├── data/                   # Not included (Kaggle dataset)
├── notebooks/              # Optional experiments (if needed)
├── src/
│   ├── detect_mask.py      # Real-time detection script
│   ├── train_model.py      # Model training script
│   ├── utils.py            # Helper functions
│   └── preprocess.py       # Image processing logic
│
├── models/
│   └── mask_detector.h5    # Trained model
│
├── results/                # Output visuals (empty if not used)
├── README.md               # Phase 3 documentation
├── requirements.txt        # Python dependencies
└── report_final.docx       # Final Phase 3 report
