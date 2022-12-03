from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import os
import csv
import numpy as np
import cv2
from sklearn import svm
import dlib

from file_functions import get_samples, scale_sample
from feature_extraction import detect_face, extract_mustache_region, extract_mouth_region, extract_nose_region, extract_right_eyebrow_region
from dlib_extraction import get_dlib_landmarks

mustache_features_model = None
mouth_open_features_model = None
big_nose_features_model = None
bushy_eyebrow_features_model = None

# Get dataset directory
DATASET_DIR = os.path.abspath('face-dataset/faces')

# Initialize dlib predictor
DLIB_PREDICTOR = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat') 

def extract_vgg_features(img, features_model):
    # prepare the image for VGG
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img[np.newaxis, :, :, :]
    # call feature extraction
    return features_model.predict(img,verbose=0)

def train_mustache():

    # We are utilizing Keras API to load the pretrained VGG16 model as our feature extractor for training SVM
    model = VGG16(weights='imagenet')

    # Choose the layer of the VGG model used to get your features (= "network embeddings")
    cnn_codes = 'fc2'

    # Loading our model that will output the network enbeddings specified by us above (instead of a classification decision)
    global mustache_features_model
    mustache_features_model = Model(inputs=model.input, outputs=model.get_layer(cnn_codes).output)

    celebA_images = []
    celebA_mustaches = []

    with open(os.path.abspath('face-dataset/list_attr_celeba.csv')) as file_obj:

        # Loop over all images in dataset
        reader_obj = csv.reader(file_obj)

        for row_number, row in enumerate(reader_obj):

            if (row_number == 0 or row_number > 400):
                continue

            # Get image path
            img_path = os.path.join(DATASET_DIR, row[0])

            # Get original and grayscale samples
            sample, sample_gray = get_samples(img_path)
            sample = scale_sample(sample)
            sample_gray = scale_sample(sample_gray)

            # Detect faces
            faces = detect_face(sample, sample_gray)

            # If no face is detected, skip to next sample
            if (len(faces) != 1):
                continue

            # Iterate through identified faces in an image
            for face in faces:

                # Get landmarks
                landmarks = get_dlib_landmarks(sample, face, DLIB_PREDICTOR)

                # Get mustache region. defined by two points
                ((x1, y1), (x2, y2)) = extract_mustache_region(landmarks)

                # Create a sample cropped around the mouth region
                cropped_sample = sample[y1:y2, x1:x2]

                # Resize the cropped sample to 224x224px
                cropped_sample = cv2.resize(cropped_sample, (224, 224), interpolation=cv2.INTER_LINEAR)

                # Add cropped image to list of images
                celebA_images.append(cropped_sample)

            # If subject has a mustache or goatee, mark as True
            if (row[17] == '1' or row[23] == '1'):
                celebA_mustaches.append(True)
            else:
                celebA_mustaches.append(False)

    # Train the model on the cropped images and whether the subjects' have a mustache
    celebA_vgg_features = mustache_features_model.predict(np.array(celebA_images), batch_size=128, workers=2, use_multiprocessing=True, verbose=1)

    # Create a classifier
    clf = svm.SVC(kernel='linear').fit(celebA_vgg_features, celebA_mustaches)

    # Return classifier
    return clf

def predict_mustache(mustache_sample, mustache_clf):

    # create transparent overlay for svm classification info
    mustache_sample = cv2.resize(mustache_sample, (224, 224), interpolation=cv2.INTER_LINEAR)
    svm_overlay = np.zeros([224,224,4], dtype=np.uint8)

    features = extract_vgg_features(mustache_sample, mustache_features_model)
    pred = mustache_clf.predict(features)
    return pred[0]

def train_mouth_open():

    # We are utilizing Keras API to load the pretrained VGG16 model as our feature extractor for training SVM
    model = VGG16(weights='imagenet')

    # Choose the layer of the VGG model used to get your features (= "network embeddings")
    cnn_codes = 'fc2'

    # Loading our model that will output the network enbeddings specified by us above (instead of a classification decision)
    global mouth_open_features_model
    mouth_open_features_model = Model(inputs=model.input, outputs=model.get_layer(cnn_codes).output)

    celebA_images = []
    celebA_mouth_open = []

    with open(os.path.abspath('face-dataset/list_attr_celeba.csv')) as file_obj:

        # Loop over all images in dataset
        reader_obj = csv.reader(file_obj)

        for row_number, row in enumerate(reader_obj):

            if (row_number == 0 or row_number > 400):
                continue

            # Get image path
            img_path = os.path.join(DATASET_DIR, row[0])

            # Get original and grayscale samples
            sample, sample_gray = get_samples(img_path)
            sample = scale_sample(sample)
            sample_gray = scale_sample(sample_gray)

            # Detect faces
            faces = detect_face(sample, sample_gray)

            # If no face is detected, skip to next sample
            if (len(faces) != 1):
                continue

            # Iterate through identified faces in an image
            for face in faces:

                # Get landmarks
                landmarks = get_dlib_landmarks(sample, face, DLIB_PREDICTOR)

                # Get mouth region. defined by two points
                ((x1, y1), (x2, y2)) = extract_mouth_region(landmarks)

                # Create a sample cropped around the mouth region
                if (y1 < y2):
                    cropped_sample = sample[y1:y2, x1:x2]
                else:
                    cropped_sample = sample[y2:y1, x1:x2]

                # Resize the cropped sample to 224x224px
                cropped_sample = cv2.resize(cropped_sample, (224, 224), interpolation=cv2.INTER_LINEAR)

                # Add cropped image to list of images
                celebA_images.append(cropped_sample)

            # If subject has their mouth slightly open, mark as True
            if (row[22] == '1'):
                celebA_mouth_open.append(True)
            else:
                celebA_mouth_open.append(False)

    # Train the model on the cropped images and whether the subjects' mouths are marked as open
    celebA_vgg_features = mouth_open_features_model.predict(np.array(celebA_images), batch_size=128, workers=2, use_multiprocessing=True, verbose=1)

    # Create a classifier
    clf = svm.SVC(kernel='linear').fit(celebA_vgg_features, celebA_mouth_open)

    # Return classifier
    return clf

def predict_mouth_open(mouth_sample, mouth_open_clf):

    # create transparent overlay for svm classification info
    mouth_sample = cv2.resize(mouth_sample, (224, 224), interpolation=cv2.INTER_LINEAR)
    svm_overlay = np.zeros([224,224,4], dtype=np.uint8)

    features = extract_vgg_features(mouth_sample, mouth_open_features_model)
    pred = mouth_open_clf.predict(features)
    return pred[0]

def train_big_nose():

    # We are utilizing Keras API to load the pretrained VGG16 model as our feature extractor for training SVM
    model = VGG16(weights='imagenet')

    # Choose the layer of the VGG model used to get your features (= "network embeddings")
    cnn_codes = 'fc2'

    # Loading our model that will output the network enbeddings specified by us above (instead of a classification decision)
    global big_nose_features_model
    big_nose_features_model = Model(inputs=model.input, outputs=model.get_layer(cnn_codes).output)

    celebA_images = []
    celebA_noses = []

    with open(os.path.abspath('face-dataset/list_attr_celeba.csv')) as file_obj:

        # Loop over all images in dataset
        reader_obj = csv.reader(file_obj)

        for row_number, row in enumerate(reader_obj):

            if (row_number == 0 or row_number > 400):
                continue

            # Get image path
            img_path = os.path.join(DATASET_DIR, row[0])

            # Get original and grayscale samples
            sample, sample_gray = get_samples(img_path)
            sample = scale_sample(sample)
            sample_gray = scale_sample(sample_gray)

            # Detect faces
            faces = detect_face(sample, sample_gray)

            # If no face is detected, skip to next sample
            if (len(faces) != 1):
                continue

            # Iterate through identified faces in an image
            for face in faces:

                # Get landmarks
                landmarks = get_dlib_landmarks(sample, face, DLIB_PREDICTOR)

                # Get nose region, defined by two points
                ((x1, y1), (x2, y2)) = extract_nose_region(landmarks)

                # Create a sample cropped around the nose region
                cropped_sample = sample[y1:y2, x1:x2]

                # Resize the cropped sample to 224x224px
                cropped_sample = cv2.resize(cropped_sample, (224, 224), interpolation=cv2.INTER_LINEAR)

                # Add cropped image to list of images
                celebA_images.append(cropped_sample)

            # If subject has a big nose, mark as True
            if (row[8] == '1'):
                celebA_noses.append(True)
            else:
                celebA_noses.append(False)

    # Train the model on the cropped images and whether the subjects' have a big nose
    celebA_vgg_features = big_nose_features_model.predict(np.array(celebA_images), batch_size=128, workers=2, use_multiprocessing=True, verbose=1)

    # Create a classifier
    clf = svm.SVC(kernel='linear').fit(celebA_vgg_features, celebA_noses)

    # Return classifier
    return clf

def predict_big_nose(nose_sample, big_nose_clf):

    # create transparent overlay for svm classification info
    nose_sample = cv2.resize(nose_sample, (224, 224), interpolation=cv2.INTER_LINEAR)
    svm_overlay = np.zeros([224,224,4], dtype=np.uint8)

    features = extract_vgg_features(nose_sample, big_nose_features_model)
    pred = big_nose_clf.predict(features)
    return pred[0]

def train_bushy_eyebrow():

    # We are utilizing Keras API to load the pretrained VGG16 model as our feature extractor for training SVM
    model = VGG16(weights='imagenet')

    # Choose the layer of the VGG model used to get your features (= "network embeddings")
    cnn_codes = 'fc2'

    # Loading our model that will output the network enbeddings specified by us above (instead of a classification decision)
    global bushy_eyebrow_features_model
    bushy_eyebrow_features_model = Model(inputs=model.input, outputs=model.get_layer(cnn_codes).output)

    celebA_images = []
    celebA_eyebrows = []

    with open(os.path.abspath('face-dataset/list_attr_celeba.csv')) as file_obj:

        # Loop over all images in dataset
        reader_obj = csv.reader(file_obj)

        for row_number, row in enumerate(reader_obj):

            if (row_number == 0 or row_number > 400):
                continue

            # Get image path
            img_path = os.path.join(DATASET_DIR, row[0])

            # Get original and grayscale samples
            sample, sample_gray = get_samples(img_path)
            sample = scale_sample(sample)
            sample_gray = scale_sample(sample_gray)

            # Detect faces
            faces = detect_face(sample, sample_gray)

            # If no face is detected, skip to next sample
            if (len(faces) != 1):
                continue

            # Iterate through identified faces in an image
            for face in faces:

                # Get landmarks
                landmarks = get_dlib_landmarks(sample, face, DLIB_PREDICTOR)

                # Get right eyebrow region, defined by two points
                ((x1, y1), (x2, y2)) = extract_right_eyebrow_region(landmarks)

                # Create a sample cropped around the eyebrow region
                cropped_sample = sample[y1:y2, x1:x2]

                # Resize the cropped sample to 224x224px
                cropped_sample = cv2.resize(cropped_sample, (224, 224), interpolation=cv2.INTER_LINEAR)

                # Add cropped image to list of images
                celebA_images.append(cropped_sample)

            # If subject has a bushy eyebrow, mark as True
            if (row[13] == '1'):
                celebA_eyebrows.append(True)
            else:
                celebA_eyebrows.append(False)

    # Train the model on the cropped images and whether the subjects' have a bushy eyebrow
    celebA_vgg_features = bushy_eyebrow_features_model.predict(np.array(celebA_images), batch_size=128, workers=2, use_multiprocessing=True, verbose=1)

    # Create a classifier
    clf = svm.SVC(kernel='linear').fit(celebA_vgg_features, celebA_eyebrows)

    # Return classifier
    return clf

def predict_bushy_eyebrow(eyebrow_sample, bushy_eyebrow_clf):

    # create transparent overlay for svm classification info
    eyebrow_sample = cv2.resize(eyebrow_sample, (224, 224), interpolation=cv2.INTER_LINEAR)
    svm_overlay = np.zeros([224,224,4], dtype=np.uint8)

    features = extract_vgg_features(eyebrow_sample, bushy_eyebrow_features_model)
    pred = bushy_eyebrow_clf.predict(features)
    return pred[0]