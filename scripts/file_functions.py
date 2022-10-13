import cv2

def get_samples(img_path):
    '''From a file path, reads an image into its original sample and grayscale sample'''

    # Read the image
    sample = cv2.imread(img_path)

    # Convert the original iamge to grayscale
    sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    return sample, sample_gray

def display_images(titles, images):
    '''Given a title and an image file, create a window displaying the image.
    Close the window when ESC is pressed.'''

    if (len(titles) != len(images)):
        print("[ERROR] The number of titles and images are not equal")
        quit()

    while True:

        # Loop through desired windows
        for i in range(len(titles)):

            # Create window
            cv2.imshow(titles[i], images[i])

        # Wait for ESC key then close window
        action = cv2.waitKey(1)
        if action & 0xFF == 27:
            break

def scale_sample(sample):
    '''Scales an sample to have a width of 640px'''

    WIDTH = 640
    scale_percent = int(WIDTH / sample.shape[1])
    height = int(sample.shape[0] * scale_percent)
    dim = (WIDTH, height)
    return cv2.resize(sample, dim)