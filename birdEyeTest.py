import cv2 as cv
import numpy as np

# Mouse callback function
def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv.imshow('Image', image)
# Load an image
image = cv.imread('photo/birdeyetest.jpg')
cv.imshow('Image', image)

# Initialize clicked points list
clicked_points = []

# Set the mouse callback function
cv.setMouseCallback('Image', click_event)

cv.waitKey(0)
cv.destroyAllWindows()

# Check if at least 4 points were clicked
if len(clicked_points) >= 4:
    # Define source and destination points for perspective transformation
    src_points = np.float32(clicked_points)
    dst_points = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])  # Adjust the destination points as needed

    # Calculate perspective transformation matrix
    matrix = cv.getPerspectiveTransform(src_points, dst_points)

    # Perform perspective transformation
    birdseye_view = cv.warpPerspective(image, matrix, (500, 500))  # Adjust the size as needed

    # Display the bird's-eye view image
    cv.imshow('Birdseye View', birdseye_view)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Please click at least 4 points on the image.")
