import cv2 as cv
import numpy as np

# Define the FaceDetector class
class FaceDetector:
    def __init__(self, model_path):
        self.face_cascade = cv.CascadeClassifier(model_path)

    def detect_faces(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

# Load pre-trained models
face_detector = FaceDetector(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_net = cv.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
age_net = cv.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

# Age and gender categories
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Open webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = face_detector.detect_faces(frame)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face ROI for age and gender models
        blob = cv.dnn.blobFromImage(face_roi, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Draw rectangle around the face
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display gender and age above the face
        label = f'{gender}, {age}'
        cv.putText(frame, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv.imshow('Face, Age, and Gender Detection', frame)

    # Exit on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
