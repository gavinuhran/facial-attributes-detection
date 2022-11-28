import dlib
import numpy as np

def get_dlib_landmarks(sample, face, predictor):
    '''Perform dlib detection that returns 68 landmarks on an identified face'''

    x = face_rectangle[0]
    y = face_rectangle[1]
    w = face_rectangle[2]
    h = face_rectangle[3]

    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  

    landmarks = np.matrix([[p.x, p.y]  
                for p in predictor(sample, dlib_rect).parts()])  

def draw_landmarks(sample, landmarks, enumerated=False):
    '''Draw the 68 detected dlib landmarks on the sample'''

    if enumerated:
        for idx, point in enumerate(landmarks):  
            pos = (point[0, 0], point[0, 1])  
            cv2.putText(image, str(idx), pos,  
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,  
                fontScale=0.4,  
                color=(0, 0, 255))  
        
            cv2.circle(image, pos, 2, color=(0, 255, 255), thickness=-1)  
    else:
        for point in landmarks:  
            pos = (point[0, 0], point[0, 1])
        
            cv2.circle(sample, pos, 2, color=(0, 255, 255), thickness=-1)