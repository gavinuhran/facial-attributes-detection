# Import libraries
import os
import cv2

# Import functions
from detect_face import detect_face
from file_functions import get_samples, display_images, scale_sample
from morphology import get_binary_sample, morph_open, morph_gradient, morph_erode
from edge_detection import canny_edge_detection
    

def main():
    '''Loops over all images in dataset, passing them into functions that 
    detect a face and analyze for facial attributes'''

    # Get dataset directory
    DATASET_DIR = os.path.abspath('face-dataset/faces')

    # Loop over all images in dataset
    for filename in os.listdir(DATASET_DIR):

        # Get image path
        img_path = os.path.join(DATASET_DIR, filename)

        # Get original and grayscale samples
        sample, sample_gray = get_samples(img_path)

        # Detect faces
        is_face_detected, face_detection = detect_face(sample, sample_gray)

        # If no face is or multiple faces are detected, skip to next sample
        if (not is_face_detected): 
            continue

        # Perform binarization on the sample
        binary_sample = get_binary_sample(sample_gray)

        # Perform canny edge detection
        canny_edges = canny_edge_detection(sample)

        # Perform morphological operations on canny edges
        morph_sample = morph_open(canny_edges, kernel_size=8)
        morph_sample = morph_erode(morph_sample, kernel_size=2, iterations=1)

        # Display samples
        titles = ["Sample", "Sample Grayscale", "Face Detection", "Binary Sample", "Canny Edges", "Morphological Sample"]
        images = [sample, sample_gray, face_detection, binary_sample, canny_edges, morph_sample]
        display_images(titles, images)


if __name__ == '__main__':
    main()