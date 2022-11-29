# Import libraries
import os
import cv2
import dlib

# Import functions
from feature_extraction import detect_face, extract_mustache_region, extract_eyebrow_region
from file_functions import get_samples, scale_sample
from dlib_extraction import get_dlib_landmarks, draw_landmarks, get_jawline_landmarks, get_right_eyebrow_landmarks, get_left_eyebrow_landmarks, get_nose_landmarks, get_right_eye_landmarks, get_left_eye_landmarks, get_lips_landmarks, get_teeth_landmarks


def main():
    '''Loops over all images in dataset, passing them into functions that 
    detect a face and analyze for facial attributes'''

    # Get dataset directory
    DATASET_DIR = os.path.abspath('face-dataset/faces')

    # Initialize dlib predictor
    DLIB_PREDICTOR = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')  

    run = True

    # Loop over all images in dataset
    for filename in os.listdir(DATASET_DIR):

        if (not run):
            cv2.destroyAllWindows()
            break

        # Get image path
        img_path = os.path.join(DATASET_DIR, filename)

        # Get original and grayscale samples
        sample, sample_gray = get_samples(img_path)
        sample = scale_sample(sample)
        sample_gray = scale_sample(sample_gray)

        # Detect faces
        faces = detect_face(sample, sample_gray)

        # If no face is detected, skip to next sample
        if (not len(faces)):
            continue

        # Iterate through identified faces in an image
        for face in faces:

            # Get landmarks
            landmarks = get_dlib_landmarks(sample, face, DLIB_PREDICTOR)
            
            # Draw nose landmarks
            draw_landmarks(
                sample, get_nose_landmarks(landmarks))

            # Draw teeth landmarks
            draw_landmarks(
                sample, get_teeth_landmarks(landmarks))

            # Draw box around mustache region
            mustache_region = extract_mustache_region(landmarks)
            cv2.rectangle(
                sample, mustache_region[0], mustache_region[1], color=(0, 255, 0), thickness=2)

            # Draw box around right eyebrow region
            eyebrow_region = extract_eyebrow_region(landmarks)
            cv2.rectangle(
                sample, eyebrow_region[0], eyebrow_region[1], color=(0, 255, 0), thickness=2)
            

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