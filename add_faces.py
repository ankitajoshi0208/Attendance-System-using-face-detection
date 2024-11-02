import cv2
import numpy as np
import pickle
import os

# Initialize face detector and video capture
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

# Load existing data or initialize empty arrays
if os.path.exists('data/faces_data.pkl') and os.path.exists('data/names.pkl'):
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    with open('data/names.pkl', 'rb') as w:
        labels = pickle.load(w)
else:
    faces = np.empty((0, 7500))
    labels = []

# Get user name for labeling
name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in detected_faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten()

        # Check for consistent shape
        if resized_img.shape[0] == faces.shape[1]:
            faces = np.append(faces, [resized_img], axis=0)
            labels.append(name)
        else:
            print("Error: Inconsistent image size. Please check your camera resolution.")

    cv2.imshow("Adding Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save updated data
with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(faces, f)
with open('data/names.pkl', 'wb') as w:
    pickle.dump(labels, w)

video.release()
cv2.destroyAllWindows()
