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