# Import libraries
import os
import cv2
import dlib
import csv

# Import functions
from feature_extraction import detect_face, extract_mustache_region, extract_right_eyebrow_region, extract_mouth_region, extract_nose_region, extract_left_eyebrow_region
from file_functions import get_samples, scale_sample
from dlib_extraction import get_dlib_landmarks
from classification import train_mustache, predict_mustache, train_mouth_open, predict_mouth_open, train_big_nose, predict_big_nose, train_bushy_eyebrow, predict_bushy_eyebrow


def main():
    '''Loops over all images in dataset, passing them into functions that 
    detect a face and analyze for facial attributes'''

    # Get dataset directory
    DATASET_DIR = os.path.abspath('face-dataset/sequestered-faces')


    '''TRAIN CLASSIFIERS FOR ATTRIBUTES To BE DETECTED'''

    # Train classifier for mustache
    print('Training mustache classifier...')
    mustache_clf = train_mustache()
    print('Completed training')

    # Train classifier for mouth open
    print('Training mouth open classifier...')
    mouth_open_clf = train_mouth_open()
    print('Completed training')

    # Train classifier for big nose
    print('Training big nose classifier...')
    big_nose_clf = train_big_nose()
    print('Completed training')

    # Train classifier for bushy eyebrow
    print('Training bushy eyebrow classifier...')
    bushy_eyebrow_clf = train_bushy_eyebrow()
    print('Completed training')

    # Initialize dlib predictor
    DLIB_PREDICTOR = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')  

    # Variable used for breaking out of main loop
    run = True

    # Loop over all images in sequestered dataset
    for filename in os.listdir(os.path.abspath('face-dataset/sequestered-faces')):

            # If loop should break, then break
            if (not run):
                cv2.destroyAllWindows()
                break

            # Get image path
            img_path = os.path.join(DATASET_DIR, filename)
            print(img_path)

            # Get original and grayscale samples, scaling to 480x640px
            sample, sample_gray = get_samples(img_path)
            sample = scale_sample(sample)
            sample_gray = scale_sample(sample_gray)

            # Detect any faces in sample
            faces = detect_face(sample, sample_gray)

            # If no face is detected, or more than 1 face is detected, skip to next sample
            if (len(faces) != 1):
                continue

            # Iterate through identified faces in an image
            for face in faces:

                # Get landmarks
                landmarks = get_dlib_landmarks(sample, face, DLIB_PREDICTOR)

                '''BIG NOSE DETECTION'''
                
                # Get nose region, defined by two points
                ((x1, y1), (x2, y2)) = extract_nose_region(landmarks)
                
                # Create a sample cropped around the nose region
                nose_sample = sample[y1:y2, x1:x2]

                # If the model predicts that the nose sample has a big nose, draw a rectangle on the nose region
                big_nose_pred = predict_big_nose(nose_sample, big_nose_clf)
                if (big_nose_pred):
                    cv2.rectangle(
                        sample, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2) 

                '''MOUTH SLIGHTLY OPEN DETECTION'''

                # Draw teeth region, defined by two points
                ((x1, y1), (x2, y2)) = extract_mouth_region(landmarks)

                # Create a sample cropped around the mouth region
                if (y1 < y2):
                    mouth_sample = sample[y1:y2, x1:x2]
                else:
                    mouth_sample = sample[y2:y1, x1:x2]

                # If the model predicts that the mustache sample has a mustache, draw a rectangle on the mustache region
                mouth_open_pred = predict_mouth_open(mouth_sample, mouth_open_clf)
                if (mouth_open_pred):
                    cv2.rectangle(
                        sample, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                '''MUSTACHE DETECTION'''

                # Get mustache region, defined by two points
                ((x1, y1), (x2, y2)) = extract_mustache_region(landmarks)

                # Create a sample cropped around the mustache region
                mustache_sample = sample[y1:y2, x1:x2]

                # If the model predicts that the mustache sample has a mustache, draw a rectangle on the mustache region
                mustache_pred = predict_mustache(mustache_sample, mustache_clf)
                if (mustache_pred):
                    cv2.rectangle(
                        sample, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2) 

                '''BUSHY EYEBROW DETECTION'''

                # Get right eyebrow region, defined by two points
                ((x1, y1), (x2, y2)) = extract_right_eyebrow_region(landmarks)

                # Get left eyebrow region, defined by two points
                ((x3, y3), (x4, y4)) = extract_left_eyebrow_region(landmarks)

                # Create a sample cropped around the eyebrow region
                eyebrow_sample = sample[y1:y2, x1:x2]

                # If the model predicts that the eyebrow region has a bushy eyebrow, draw a rectangle over both eyebrows
                eyebrow_pred = predict_bushy_eyebrow(eyebrow_sample, bushy_eyebrow_clf)
                if (eyebrow_pred):
                    cv2.rectangle(
                        sample, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2) 
                    cv2.rectangle(
                        sample, (x3, y3), (x4, y4), color=(0, 0, 255), thickness=2) 
                

            # Display sample
            cv2.putText(
                sample, filename, (5, 25), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.8, color=(0, 255, 0))
            cv2.imshow("Landmarks found", sample)
            key = cv2.waitKey(0)
            if key == 27:
                run = False
                break

if __name__ == '__main__':
    main()