import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os

def detect_face(sample, sample_gray):

    # Load HaarCascase model
    HAARCASCADE_PATH = os.path.abspath('models/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    faces = faceCascade.detectMultiScale(
        sample_gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # Print if face is detected
    print(f'[INFO] {len(faces)} face(s) detected')

    # Skip images with no face or multiple faces
    if (len(faces) != 1):
        return False, None

    # Duplicate sample into new sample
    face_detection = sample.copy()

    # Draw rectangle on face
    for (x, y, w, h) in faces:
        cv2.rectangle(face_detection, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return True, face_detection

    # # Binarize the image using Otsu's method
    # ret1, binary_sample = cv2.threshold(src=sample_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # # Morphological operations to isolate the face shape
    # kernel = np.ones((5,5),np.uint8)
    # morph_sample = cv2.morphologyEx(binary_sample, cv2.MORPH_OPEN, kernel)
    # morph_sample = cv2.morphologyEx(morph_sample, cv2.MORPH_GRADIENT, kernel)