import cv2 as cv
import numpy as np
from ultralytics import YOLO
from sort import Sort  
import cvzone  

model = YOLO("yolov8m.pt")
classNames = model.names

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

cap = cv.VideoCapture("traffic_final.mp4")
print("Video opened:", cap.isOpened())


line = [100, 500, 1000, 500] #---> line for counting (x1,y1) and (x2,y2)

totalCount = []

# writer = None

while True:
    success, frame = cap.read()
    frame = cv.resize(frame,(1080,720))
    if not success:
        break

    results = model(frame, stream=True, classes=[2, 3, 5, 7]) #---> specific classes(cars,bus,truck,motorbikes)
    detections = np.empty((0, 5)) # [x1, y1, x2, y2, confidence]

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.5:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

               
                # label = f"{classNames[cls]} {conf:.2f}"
                # cvzone.putTextRect(frame, label, (x1, y1), scale=0.7, thickness=1)
                # cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    resultsTracker = tracker.update(detections)

    # counting line
    cv.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

        # to check if the vehicle crosses the line
        if line[0] < cx < line[2] and (line[1] - 15) < cy < (line[1] + 15):
            if id not in totalCount:
                totalCount.append(id)
                cv.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

        
        cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=5, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f'ID: {int(id)}', (x1, y1 - 10), scale=1, thickness=2)

   
    cv.putText(frame, f'Count: {len(totalCount)}', (50, 50),cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Initialize video writer once
    # if writer is None:
    #     fourcc = cv.VideoWriter_fourcc(*'mp4v')
    #     height, width = frame.shape[:2]
    #     writer = cv.VideoWriter('output.mp4', fourcc, 20, (width, height))

    # writer.write(frame)
    cv.imshow("Vehicle Tracking", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# writer.release()
cv.destroyAllWindows()
