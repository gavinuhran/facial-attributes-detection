import cv2

def canny_edge_detection(sample):
    '''Perform canny edge detection on the sample, using preset threshold values'''

    edges = sample.copy()
    return cv2.Canny(edges, 245, 255) #, apertureSize=3, L2gradient=True)