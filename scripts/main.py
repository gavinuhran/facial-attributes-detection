# Import libraries
import os
import cv2
import dlib

# Import functions
from feature_extraction import detect_face, detect_eyes, detect_mouth, is_positioning_weird
from file_functions import get_samples, display_images, scale_sample
from morphology import get_binary_sample, morph_open, morph_gradient, morph_erode
from edge_detection import canny_edge_detection
    

def main():
    '''Loops over all images in dataset, passing them into functions that 
    detect a face and analyze for facial attributes'''

    # Get dataset directory
    DATASET_DIR = os.path.abspath('face-dataset/faces')

    # Initialize dlib predictor
    DLIB_PREDICTOR = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')  

    # Loop over all images in dataset
    for filename in os.listdir(DATASET_DIR):

        # Get image path
        img_path = os.path.join(DATASET_DIR, filename)

        # Get original and grayscale samples
        sample, sample_gray = get_samples(img_path)

        # List of rectangles to be drawn, highlighting extracted features
        feature_rectangles = []

        # Detect faces
        is_face_detected, face_rectangle = detect_face(sample, sample_gray)

        # If no face is detected, skip to next sample
        if (not is_face_detected): 
            continue
        else:
            feature_rectangles.append(face_rectangle)

        # Detect eyes
        are_eyes_detected, eye_rectangles = detect_eyes(sample, sample_gray)

        # If no eyes are detected, skip to next sample
        if (not are_eyes_detected):
            continue
        else:
            feature_rectangles.extend(eye_rectangles)

        # Detect mouth
        is_mouth_detected, mouth_rectangle = detect_mouth(sample, sample_gray)

        # If no mouth is detected, skip to next sample
        if (not is_mouth_detected):
            continue
        else:
            feature_rectangles.append(mouth_rectangle)

        # Check that the positioning of the eyes and mouth all make sense
        if (is_positioning_weird(feature_rectangles)):
            continue

        # Draw rectangles on sample
        extracted_features = sample.copy()
        for (x, y, w, h) in feature_rectangles:
            cv2.rectangle(extracted_features, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display samples
        # titles = ["Sample", "Sample Grayscale", "Face Detection", "Binary Sample", "Canny Edges", "Morphological Sample"]
        # images = [sample, sample_gray, left_eye, binary_sample, canny_edges, morph_sample]
        # display_images(titles, images)
        display_images(['Features'], [extracted_features])


if __name__ == '__main__':
    main()