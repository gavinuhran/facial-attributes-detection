# Import libraries
import os
import cv2

# Import functions
from detect_face import detect_face
from file_functions import get_samples, display_images
    

def main():
    '''Loops over all images in dataset, passing them into functions that 
    detect a face and analyze for facial attributes'''

    # Loop over all images in dataset
    for i in range(1,10):

        # Get image path
        img_path = os.path.abspath(f'face-dataset/faces/00000{i}.jpg')

        # Get original and grayscale samples
        sample, sample_gray = get_samples(img_path)

        # Detect faces
        is_face_detected, face_detection = detect_face(sample, sample_gray)

        # If no face is or multiple faces are detected, skip to next sample
        if (not is_face_detected): 
            continue

        # Display samples
        titles = ["Sample", "Sample Grayscale", "Face Detection"]
        images = [sample, sample_gray, face_detection]
        display_images(titles, images)


if __name__ == '__main__':
    main()