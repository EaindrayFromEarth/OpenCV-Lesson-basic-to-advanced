import cv2 as cv
import dlib
import numpy as np

# Load the pre-trained face detection cascade classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the facial landmarks predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the headphone image with a transparent background
headphone_image = cv.imread('photo\headphone.png', cv.IMREAD_UNCHANGED)

# Create a VideoCapture object to access the webcam (0 is usually the default camera)
cap = cv.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Create a named window for displaying the webcam feed
cv.namedWindow('Webcam Face Detection', cv.WINDOW_NORMAL)
cv.resizeWindow('Webcam Face Detection', 800, 600)

# Initialize variables for creative effects
save_face_counter = 0
filter_active = False

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

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get facial landmarks
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = predictor(gray_frame, rect)

        # Convert landmarks to a NumPy array
        landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        # Calculate the position for the headphone
        headphone_x = landmarks[30][0] - w // 2  # Adjust the coordinates as needed
        headphone_y = landmarks[1][1] - h // 2

        # Resize the headphone image to fit the face
        resized_headphone = cv.resize(headphone_image, (w, h))

        # Overlay the headphone image onto the detected face
        for c in range(0, 3):
            frame[y + headphone_y:y + headphone_y + h, x + headphone_x:x + headphone_x + w, c] = \
                frame[y + headphone_y:y + headphone_y + h, x + headphone_x:x + headphone_x + w, c] * \
                (1 - resized_headphone[:, :, 3] / 255.0) + \
                resized_headphone[:, :, c] * (resized_headphone[:, :, 3] / 255.0)

        # You can perform creative effects here, e.g., apply filters to the detected face
        if filter_active:
            # Sample creative effect: draw a pink mask on the face
            mask_color = (255, 192, 203)  # Pink color
            cv.fillPoly(frame, [landmarks[48:60]], mask_color)

        # Save the detected face as an image
        face_roi = frame[y:y + h, x:x + w]
        save_face_counter += 1
        cv.imwrite(f'detected_faces/face_{save_face_counter}.png', face_roi)

    # Display the frame with detected faces and the effects
    cv.imshow('Webcam Face Detection', frame)

    # Check for user input
    key = cv.waitKey(1) & 0xFF

    # Press 's' to save the creative effect
    if key == ord('s'):
        filter_active = not filter_active

    # Press 'q' to quit the application
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
