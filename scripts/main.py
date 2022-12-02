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
    DATASET_DIR = os.path.abspath('face-dataset/faces')


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

    # Variables for providing statistics on the classifiers' performances
    nose_fp = 0  # fp -> false positives
    nose_fn = 0  # fn -> false negatives
    nose_correct = 0
    mouth_fp = 0
    mouth_fn = 0
    mouth_correct = 0
    mustache_fp = 0
    mustache_fn = 0
    mustache_correct = 0
    eyebrow_fp = 0
    eyebrow_fn = 0
    eyebrow_correct = 0
    total = 0

    # Open dataset for testing
    with open(os.path.abspath('face-dataset/list_attr_celeba.csv')) as file_obj:

        # Loop over all images in dataset
        reader_obj = csv.reader(file_obj)
        for row_number, row in enumerate(reader_obj):

            # If loop should break, then break
            if (not run):
                cv2.destroyAllWindows()
                break
            
            # Skip first row, limit testing partition to last 100 images
            if (row_number == 0 or row_number <= 400):
                continue

            # Get image filename from first field in row
            filename = row[0]            

            # Get image path
            img_path = os.path.join(DATASET_DIR, filename)

            # Get original and grayscale samples, scaling to 480x640px
            sample, sample_gray = get_samples(img_path)
            sample = scale_sample(sample)
            sample_gray = scale_sample(sample_gray)

            # Detect any faces in sample
            faces = detect_face(sample, sample_gray)

            # If no face is detected, or more than 1 face is detected, skip to next sample
            if (len(faces) != 1):
                continue

            total += 1

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
                if (bool(int(row[8]) + 1) == big_nose_pred):
                    nose_correct += 1
                elif (bool(int(row[8]) + 1) and not big_nose_pred):
                    nose_fn += 1
                else:
                    nose_fp += 1
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
                if (bool(int(row[22]) + 1) == mouth_open_pred):
                    mouth_correct += 1
                elif (bool(int(row[22]) + 1) and not mouth_open_pred):
                    mouth_fn += 1
                else:
                    mouth_fp += 1
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
                if (bool(int(row[23]) + 1) == mustache_pred):
                    mustache_correct += 1
                elif (bool(int(row[23]) + 1) and not mustache_pred):
                    mustache_fn += 1
                else:
                    mustache_fp += 1
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
                if (bool(int(row[13]) + 1) == eyebrow_pred):
                    eyebrow_correct += 1
                elif (bool(int(row[13]) + 1) and not eyebrow_pred):
                    eyebrow_fn += 1
                else:
                    eyebrow_fp += 1
                if (eyebrow_pred):
                    cv2.rectangle(
                        sample, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2) 
                    cv2.rectangle(
                        sample, (x3, y3), (x4, y4), color=(0, 0, 255), thickness=2) 
                

            # Display sample
            # cv2.putText(
            #     sample, filename, (5, 25), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.8, color=(0, 255, 0))
            # cv2.imshow("Landmarks found", sample)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     run = False
            #     break

    '''PRINT OUT DETAILED STATISTICS ON THE CLASSIFIERS' PERFORMANCES'''
    print('-------------------------------------------------------------')
    print(f'Total subjects:       {total}\n')    
    print('BIG NOSE ANALYSIS')
    print(f'Correctly predicted:  {(nose_correct/total) * 100.0}')
    print(f'False postives:       {(nose_fp/total) * 100.0}')
    print(f'False negatives:      {(nose_fn/total) * 100.0}')
    print('MOUTH SLIGHTLY OPEN ANALYSIS')
    print(f'Correctly predicted:  {(mouth_correct/total) * 100.0}')
    print(f'False postives:       {(mouth_fp/total) * 100.0}')
    print(f'False negatives:      {(mouth_fn/total) * 100.0}')
    print('MUSTACHE ANALYSIS')
    print(f'Correctly predicted:  {(mustache_correct/total) * 100.0}')
    print(f'False postives:       {(mustache_fp/total) * 100.0}')
    print(f'False negatives:      {(mustache_fn/total) * 100.0}')
    print('BUSHY EYEBROW ANALYSIS')
    print(f'Correctly predicted:  {(eyebrow_correct/total) * 100.0}')
    print(f'False postives:       {(eyebrow_fp/total) * 100.0}')
    print(f'False negatives:      {(eyebrow_fn/total) * 100.0}')
    print('-------------------------------------------------------------')


if __name__ == '__main__':
    main()