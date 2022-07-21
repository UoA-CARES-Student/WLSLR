from pickle import TRUE
import queue
from cv2 import FONT_HERSHEY_COMPLEX, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_TRIPLEX
import numpy as np
import cv2 as cv
import predict
import collections
from threading import Thread, active_count, current_thread

frames = collections.deque([])

cap = cv.VideoCapture(0)
vid_num = 0

fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter(f"test{vid_num}.mp4", fourcc, 20.0, (640,480))

preds_queue = queue.Queue(maxsize=2)
pred_text = "none"

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    predict_thread = Thread(target=predict.predict_from_mp4, args=(f"test{vid_num}.mp4",preds_queue))

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if not preds_queue.empty():
        pred = preds_queue.get()
        if float(pred[1]) > 0.8:
            pred_text = pred[0]
        else:
            pred_text = "none"
    # Our operations on the frame come here
    cv.putText(img=frame, text=pred_text, org=(10, 40), fontFace=FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 0, 0))
    
    # Display the resulting frame
    cv.imshow('frame', frame)
    frames.appendleft(frame)

    if(len(frames) > 60 and active_count()<2):
        vid_num += 1
        out = cv.VideoWriter(f"test{vid_num}.mp4", fourcc, 20.0, (640,480))
        predict_thread.start()
        frames.clear()
    
    if(len(frames) <= 60):
        out.write(frame)

    while(len(frames) > 60):
        frames.pop()
    
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()