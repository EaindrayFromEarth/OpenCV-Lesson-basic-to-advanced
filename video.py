import cv2 as cv
import numpy as np

# 1. Open a video file for processing
cap = cv.VideoCapture('video.mp4')

# 2. Loop through the video frames
while cap.isOpened():
    # 3. Read a frame from the video
    ret, frame = cap.read()

    # 4. Display the frame
    cv.imshow('Frame', frame)

    # 5. Check for the 'q' key press to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Release the video capture
cap.release()

# 7. Close all OpenCV windows
cv.destroyAllWindows()
