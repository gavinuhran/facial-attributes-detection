import cv2
import numpy as np

def get_binary_sample(sample_gray):
    '''Returns a binary sample from a grayscale sample'''

    # Binarize the image using Otsu's method
    _, binary_sample = cv2.threshold(src=sample_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return binary_sample

def morph_erode(sample, kernel_size, iterations):
    '''Returns a sample after an morphological erosion operation'''

    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.erode(sample, kernel, iterations=iterations)

def morph_dilate(sample, kernel_size, iterations):
    '''Returns a sample after an morphological dilation operation'''

    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.dilate(sample, kernel, iterations=iterations)

def morph_open(sample, kernel_size):
    '''Returns a sample after an morphological opening operation'''

    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.morphologyEx(sample, cv2.MORPH_OPEN, kernel)

def morph_open(sample, kernel_size):
    '''Returns a sample after an morphological closing operation'''

    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.morphologyEx(sample, cv2.MORPH_CLOSE, kernel)

def morph_gradient(sample, kernel_size):
    '''Returns a sample after an morphological gradient operation'''

    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.morphologyEx(sample, cv2.MORPH_GRADIENT, kernel)

def morph_tophat(sample, kernel_size):
    '''Returns a sample after an morphological top hat operation'''

    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.morphologyEx(sample, cv2.MORPH_TOPHAT, kernel)

def morph_blackhat(sample, kernel_size):
    '''Returns a sample after an morphological black hat operation'''

    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.morphologyEx(sample, cv2.MORPH_BLACKHAT, kernel)
    