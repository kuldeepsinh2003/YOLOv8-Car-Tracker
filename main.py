import cv2
import numpy as np
from sympy import limit
from ultralytics import YOLO
import cvzone
import math
from sort import *



# Load video
#cap = cv2.VideoCapture("data/f2.mp4")
cap = cv2.VideoCapture(0)
# Load YOLO model
model = YOLO('..\\yolo_weights\\yolov8n.pt')

#tracking
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits=[295,250,980,250]
totalcount=[]

# COCO classes

# COCO class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


# Define ROI polygon (example points, you must adjust to your road area)
roi_points = np.array([
    [150, 70],   # top-left
    [1100, 70],   # top-right
    [1100, 620],   # bottom-right
    [150, 620]    # bottom-left
])

# Create mask from polygon
def create_polygon_mask(frame, points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)   # single channel
    cv2.fillPoly(mask, [points], 255)                  # fill ROI white
    return mask

while True:
    success, frame = cap.read()
    if not success:
        break
    

    # Create ROI mask
    mask = create_polygon_mask(frame, roi_points)

    # Apply mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Run YOLO only inside ROI
    results = model(masked_frame, stream=True)
    detections=np.empty((0,5))
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(masked_frame, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            if currentclass=="car" or currentclass=="bus" or currentclass=="truck":     
               '''cvzone.putTextRect(masked_frame, f'{currentclass} {conf}',
                               (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=3)'''  
              # cvzone.cornerRect(masked_frame,(x1,y1,w,h),l=9,rt=5)
               currentarray=np.array([x1,y1,x2,y2,conf])
               detections=np.vstack((detections,currentarray))
           
    resulttracker=tracker.update(detections)
    cv2.line(masked_frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    for result in resulttracker:
        x1,y1,x2,y2,id= result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(masked_frame,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(masked_frame, f'{int(id)}',
                               (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=3)
        
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(masked_frame,(cx,cy),5,(255,0,255),cv2.FILLED)
        if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[1]+20:
            if totalcount.count(id)==0:
               totalcount.append(id)
               cv2.line(masked_frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)

    #cvzone.putTextRect(masked_frame, f'count :{len(totalcount)}',(50, 50))
    cv2.putText(masked_frame,str(len(totalcount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    # Show output
    cv2.imshow("Masked ROI", masked_frame)
     

   
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

