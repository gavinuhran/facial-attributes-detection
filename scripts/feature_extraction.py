import cv2
import os

def detect_face(sample, sample_gray):

    # Load HaarCascade model
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

    return faces

def detect_eyes(sample, sample_gray):

    # Load HaarCascade model
    HAARCASCADE_EYE_PATH = os.path.abspath('models/haarcascade_eye.xml')
    eye_cascade = cv2.CascadeClassifier(HAARCASCADE_EYE_PATH)
    eyes = eye_cascade.detectMultiScale(
        sample_gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(5, 5),
        maxSize=(30,30)
    )

    # Print if  eye is detected
    # print(f'[INFO] {len(eyes)}  eye(s) detected')

    # Skip images that don't have two eyes
    if (len(eyes) != 2):
        return False, None

    # Check if the two eyes detected are similar in y-value
    if (abs(eyes[0][1] - eyes[1][1]) > 20):
        return False, None

    return True, eyes

def detect_mouth(sample, sample_gray):

    # Load HaarCascade model
    HAARCASCADE_SMILE_PATH = os.path.abspath('models/haarcascade_smile.xml')
    smile_cascade = cv2.CascadeClassifier(HAARCASCADE_SMILE_PATH)
    mouths = smile_cascade.detectMultiScale(
        sample_gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(5,5),
        maxSize=(50, 40)
    )

    # Skip images with no mouth or multiple mouths
    if (len(mouths) != 1):
        return False, None

    return True, mouths[0]

def is_positioning_weird(feature_rectangles):

    # Check that the both eyes are within the face
    # Eye 1 - horizontal
    if (feature_rectangles[1][0] < feature_rectangles[0][0] and feature_rectangles[1][0] + feature_rectangles[1][2] > feature_rectangles[0][0] + feature_rectangles[0][2]):
        return True
    # Eye 1 - vertical
    if (feature_rectangles[1][1] < feature_rectangles[0][1] and feature_rectangles[1][1] + feature_rectangles[1][3] > feature_rectangles[0][1] + feature_rectangles[0][3]):
        return True
    # Eye 2 - horizontal
    if (feature_rectangles[2][0] < feature_rectangles[0][0] and feature_rectangles[2][0] + feature_rectangles[2][2] > feature_rectangles[0][0] + feature_rectangles[0][2]):
        return True
    # Eye 2 - vertical
    if (feature_rectangles[2][1] < feature_rectangles[0][1] and feature_rectangles[2][1] + feature_rectangles[2][3] > feature_rectangles[0][1] + feature_rectangles[0][3]):
        return True

    # Check that the mouth is within the face
    # Horizontal
    if (feature_rectangles[3][0] < feature_rectangles[0][0] and feature_rectangles[3][0] + feature_rectangles[3][2] > feature_rectangles[0][0] + feature_rectangles[0][2]):
        return True
    # Vertical
    if (feature_rectangles[3][1] < feature_rectangles[0][1] and feature_rectangles[3][1] + feature_rectangles[3][3] > feature_rectangles[0][1] + feature_rectangles[0][3]):
        return True

    # Check that mouth is located below both eyes
    if (feature_rectangles[3][1] < feature_rectangles[1][1] or feature_rectangles[3][1] < feature_rectangles[2][1]):
        return True

    return False

def extract_mustache_region(landmarks):
    
    x1 = landmarks[48][0,0]
    y1 = landmarks[33][0,1]
    x2 = landmarks[54][0,0]
    y2 = landmarks[52][0,1]

    # ([x1,y1], [x2,y2]) format
    return ((x1, y1), (x2, y2))

def extract_right_eyebrow_region(landmarks):

    x1 = landmarks[17][0,0]
    y1 = min(landmarks[17:22], key=lambda x: x[0,1])[0,1]
    x2 = landmarks[21][0,0]
    y2 = max(landmarks[17:22], key=lambda x: x[0,1])[0,1]

    # ([x1,y1], [x2,y2]) format
    return ((x1, y1), (x2, y2))

def extract_left_eyebrow_region(landmarks):

    x1 = landmarks[22][0,0]
    y1 = min(landmarks[22:27], key=lambda x: x[0,1])[0,1]
    x2 = landmarks[26][0,0]
    y2 = max(landmarks[22:27], key=lambda x: x[0,1])[0,1]

    # ([x1,y1], [x2,y2]) format
    return ((x1, y1), (x2, y2))

def extract_mouth_region(landmarks):
    
    x1 = landmarks[48][0,0]
    y1 = landmarks[50][0,1]
    x2 = landmarks[54][0,0]
    y2 = landmarks[57][0,1]

    # ([x1,y1], [x2,y2]) format
    return ((x1, y1), (x2, y2))

def extract_nose_region(landmarks):

    x1 = landmarks[31][0,0] - 15
    y1 = landmarks[27][0,1]
    x2 = landmarks[35][0,0] + 15
    y2 = landmarks[33][0,1]

    # ([x1,y1], [x2,y2]) format
    return ((x1, y1), (x2, y2))