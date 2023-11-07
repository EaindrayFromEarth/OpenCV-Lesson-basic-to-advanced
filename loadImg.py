import cv2 as cv

# Load image
image = cv.imread('photo/myPhoto.jpg')

# Check if the image is loaded successfully
if image is not None:
    # Create a named window with a custom size
    cv.namedWindow('Image', cv.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
    cv.resizeWindow('Image', 800, 800)  # Set your desired window size (width, height)

    # Display the original image
    cv.imshow('Image', image)

    # Wait for a key event and then close the window
    cv.waitKey(0)
    cv.destroyAllWindows()  # Close the image window
else:
    print("Could not load the image. Please check the file path.")

# Define values for drawing
x1, y1, x2, y2 = 100, 100, 200, 200  # Line coordinates
x, y, w, h = 150, 150, 100, 100  # Rectangle coordinates
radius = 50  # Circle radius
font = cv.FONT_HERSHEY_SIMPLEX  # Font for text

# Draw shapes and apply image processing to the loaded image
# 2. Draw a blue line
cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 3. Draw a green rectangle
cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 4. Draw a filled red circle
cv.circle(image, (x, y), radius, (0, 0, 255), -1)

# 5. Draw text
cv.putText(image, 'Text', (x, y), font, 1, (255, 255, 255), 2)

# 6. Convert the image to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray_image)

# 7. Apply Gaussian Blur
kernel_size = 5
blurred_image = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
cv.imshow('Blur', blurred_image)

# 8. Perform edge detection using Canny
threshold1, threshold2 = 30, 70
edges = cv.Canny(blurred_image, threshold1, threshold2)
cv.imshow('Canny Edges', edges)

# 9. Apply image thresholding
threshold_value = 127
max_value = 255
_, thresholded = cv.threshold(gray_image, threshold_value, max_value, cv.THRESH_BINARY)

# Load a face detection cascade classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default (1).xml')

# Dilating the image
dilated = cv.dilate(edges, (7,7), iterations=3)
cv.imshow('Dilated', dilated)
# Eroding
eroded = cv.erode(dilated, (7,7), iterations=3)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(image, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = image[50:200, 200:400]
cv.imshow('Cropped', cropped)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Create a named window with a custom size for the modified image
cv.namedWindow('Modified Image', cv.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
cv.resizeWindow('Modified Image', 800, 800)  # Set your desired window size (width, height)

# Display the modified image with drawn shapes and detected faces
cv.imshow('Modified Image', image)

# Wait for a key event and then close the window
cv.waitKey(0)
cv.destroyAllWindows()  # Close the image window
