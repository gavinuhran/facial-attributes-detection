import cv2
import os

def detect_face(sample, sample_gray):

    # Load HaarCascase model
    HAARCASCADE_FACE_PATH = os.path.abspath('models/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_FACE_PATH)
    faces = face_cascade.detectMultiScale(
        sample_gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # Print if face is detected
    # print(f'[INFO] {len(faces)} face(s) detected')

    # Skip images with no face or multiple faces
    if (len(faces) != 1):
        return False, None

    # Duplicate sample into new sample
    face_detection = sample.copy()

    # Draw rectangle on face
    for (x, y, w, h) in faces:
        cv2.rectangle(face_detection, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return True, face_detection
