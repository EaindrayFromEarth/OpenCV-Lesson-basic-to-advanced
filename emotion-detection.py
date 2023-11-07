import cv2 as cv

# Load pre-trained Haar cascades for face detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default (1).xml')

# Create a VideoCapture object to access the webcam
cap = cv.VideoCapture(0)

def overlay_text(frame, x, y, w, h, emotion):
    if emotion == "smile":
        text = "😄"  # You can replace this with any text
    elif emotion == "frown":
        text = "☹️"  # You can replace this with any text
    else:
        text = "😐"  # You can replace this with any text

    # Calculate the position to overlay the text
    text_x = x + w // 2
    text_y = y - h // 4

    cv.putText(frame, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read a frame from the webcam.")
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Emotion detection (replace this with a more advanced model)
        # For simplicity, we use a rule-based approach here
        # You can replace this with an emotion detection model
        if h > w:
            emotion = "frown"  # Rule-based emotion detection (e.g., frown if height > width)
        else:
            emotion = "smile"

        overlay_text(frame, x, y, w, h, emotion)

    cv.imshow('Webcam with Emojis', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
