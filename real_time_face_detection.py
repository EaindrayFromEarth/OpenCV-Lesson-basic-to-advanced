import cv2 as cv

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default (1).xml')

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected faces
    cv.imshow('Face Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
