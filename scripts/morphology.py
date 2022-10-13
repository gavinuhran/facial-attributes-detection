import cv2

def get_binary_sample(sample_gray):
    '''Returns a binary sample from a grayscale sample'''

    # Binarize the image using Otsu's method
    _, binary_sample = cv2.threshold(src=sample_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return binary_sample