import cv2 as cv
import numpy as np

# Load the pre-trained face detection cascade classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default (1).xml')




# Load the emoji image with transparency (alpha channel)
emoji = cv.imread('photo\cool-emoji.png', cv.IMREAD_UNCHANGED)

# Create a VideoCapture object to access the webcam (0 is usually the default camera)
cap = cv.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Create a named window for displaying the webcam feed
cv.namedWindow('Webcam Face Detection', cv.WINDOW_NORMAL)
cv.resizeWindow('Webcam Face Detection', 800, 600)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read a frame from the webcam.")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Display the count of detected faces
    face_count = len(faces)
    cv.putText(frame, f'Faces Detected: {face_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Adjust the size and position of the emoji overlay on detected faces
    for (x, y, w, h) in faces:
        # Calculate the scaling factor based on the width of the face
        scale_factor = w / emoji.shape[1]

        # Resize the emoji with the calculated scale factor
        emoji_resized = cv.resize(emoji, (int(scale_factor * emoji.shape[1]), int(scale_factor * emoji.shape[0])))

        # Calculate the position for overlaying the emoji
        x1 = x  # Start from the left edge of the face
        y1 = max(y - int((emoji_resized.shape[0] - h) / 2), 0)

        x2 = min(x1 + emoji_resized.shape[1], frame.shape[1])
        y2 = min(y1 + emoji_resized.shape[0], frame.shape[0])

        # Overlay the emoji on the frame
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = frame[y1:y2, x1:x2, c] * (1 - emoji_resized[:, :, 3] / 255.0) + \
                                     emoji_resized[:, :, c] * (emoji_resized[:, :, 3] / 255.0)

    # Display the frame with detected faces and adjusted emojis
    cv.imshow('Webcam Face Detection', frame)

    # Check for the 'q' key to quit the application
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
