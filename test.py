import cv2 as cv

# Load the video
video = cv.VideoCapture(r'D:\AMRITA\SEMESTER 5\openCV\myprojects\ObjectTracking\vehicleTrackingSample\traffic.mp4')

while True:
    ret, frame = video.read()

    # Break the loop if video ends or cannot be read
    if not ret:
        break

    # Resize the frame to 720x1080
    frame = cv.resize(frame, (1080,720))

    # Draw a green line on the frame
    cv.line(frame, (100,500), (1000, 500), (0, 255, 0), 2)

    # Display the frame
    cv.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy windows
video.release()
cv.destroyAllWindows()
