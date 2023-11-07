import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('photo/myPhoto.jpg')

if image is not None:
    # Define values for drawing
    x1, y1, x2, y2 = 100, 100, 200, 200  # Line coordinates
    x, y, w, h = 150, 150, 100, 100  # Rectangle coordinates
    radius = 50  # Circle radius
    font = cv.FONT_HERSHEY_SIMPLEX  # Font for text

    # Define the remaining two points for the source quadrilateral
    x3, y3 = 200, 200
    x4, y4 = 100, 200

    # Define destination points (dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4)
    dx1, dy1 = 50, 50
    dx2, dy2 = 250, 50
    dx3, dy3 = 250, 250
    dx4, dy4 = 50, 250

    # Define source and destination points for the perspective transformation
    src_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    dst_points = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])

    # Calculate the perspective transformation matrix
    matrix = cv.getPerspectiveTransform(src_points, dst_points)

    # Get the original image dimensions
    height, width = image.shape[0], image.shape[1]

    # Apply the perspective transformation
    warped_image = cv.warpPerspective(image, matrix, (width, height))

    # Create named windows and set their sizes
    cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
    cv.resizeWindow('Original Image', 800, 800)
    cv.namedWindow('Warped Image', cv.WINDOW_NORMAL)
    cv.resizeWindow('Warped Image', 800, 800)

    # Display the original and transformed images
    cv.imshow('Original Image', image)
    cv.imshow('Warped Image', warped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Could not load the image. Please check the file path.")
