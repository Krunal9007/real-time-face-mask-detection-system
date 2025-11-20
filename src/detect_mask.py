import cv2
import argparse
import os
from tensorflow.keras.models import load_model
from utils import load_image_for_model

def main():
    model_path = "/Mask-Detection-System/models/mask_detector.h5"

    if not os.path.exists(model_path):
        print("Model not found. Train the model first.")
        return

    model = load_model(model_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to access webcam.")
        return

    print("Press 'q' to exit webcam detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            img = load_image_for_model(roi)

            pred = model.predict(img)[0][0]
            label = "Mask" if pred < 0.5 else "No Mask"
            color = (0,255,0) if label == "Mask" else (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
