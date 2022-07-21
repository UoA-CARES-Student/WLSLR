from cv2 import FONT_HERSHEY_COMPLEX, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_TRIPLEX
import numpy as np
import cv2 as cv

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter("test.mp4", fourcc, 20.0, (640,480))

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    cv.putText(img=frame, text="TEST TEST TEXT", org=(10, 40), fontFace=FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 0, 0))
    # Display the resulting frame
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()