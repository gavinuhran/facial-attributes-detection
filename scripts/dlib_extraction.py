import dlib
import numpy as np
import cv2

def get_dlib_landmarks(sample, face, predictor):
    '''Perform dlib detection that returns 68 landmarks on an identified face'''

    (x, y, w, h) = face

    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  

    landmarks = np.matrix([[p.x, p.y]  
                for p in predictor(sample, dlib_rect).parts()]) 

    return landmarks 

def draw_landmarks(sample, landmarks, enumerated=False):
    '''Draw the 68 detected dlib landmarks on the sample'''

    if enumerated:
        for idx, point in enumerate(landmarks):  
            pos = (point[0, 0], point[0, 1])  
            cv2.putText(sample, str(idx), pos,  
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,  
                fontScale=0.4,  
                color=(0, 0, 255))  
        
            cv2.circle(sample, pos, 2, color=(0, 255, 255), thickness=-1)  
    else:
        for point in landmarks:  
            pos = (point[0, 0], point[0, 1])
        
            cv2.circle(sample, pos, 2, color=(0, 255, 255), thickness=-1)

def get_jawline_landmarks(landmarks):
    return landmarks[0:17]

def get_right_eyebrow_landmarks(landmarks):
    return landmarks[17:22]

def get_left_eyebrow_landmarks(landmarks):
    return landmarks[22:27]

def get_nose_landmarks(landmarks):
    return landmarks[27:36]

def get_right_eye_landmarks(landmarks):
    return landmarks[36:42]

def get_left_eye_landmarks(landmarks):
    return landmarks[42:48]

def get_lips_landmarks(landmarks):
    return landmarks[48:60]

def get_teeth_landmarks(landmarks):
    return landmarks[60:68]
