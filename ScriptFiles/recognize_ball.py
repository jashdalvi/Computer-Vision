import cv2
import numpy as np
import imutils

weights_file = "../data/models/yolov3.weights"
cfg_file = "../data/models/yolov3.cfg"
class_file = "../data/models/coco.names"

with open(class_file) as f:
    classes = f.read().rstrip('\n').split('\n')


net = cv2.dnn.readNetFromDarknet(cfg_file,weights_file)

out_layers = net.getUnconnectedOutLayersNames()

ball_id = classes.index("sports ball")
ok = False

cap = cv2.VideoCapture("../data/videos/soccer-ball.mp4")

count = 0

while cap.isOpened():

    ret,frame = cap.read()

    start_time = cv2.getTickCount()

    if not ret:
        break
        
    frame = imutils.resize(frame,height = 500)
    height_frame , width_frame = frame.shape[:2]
    if (not ok) and count % 50 == 0:
        blob = cv2.dnn.blobFromImage(frame,1/255, (416, 416), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outputs = net.forward(out_layers)
        boxes = []
        confidences = []
        for output in outputs:
            for detection in output:
                if detection[4] > 0.5:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == ball_id:
                        center_x = int(detection[0] * width_frame)
                        center_y = int(detection[1] * height_frame)
                        w = int(detection[2]* width_frame)
                        h = int(detection[3]*height_frame)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)
        if len(indices) > 0:
            for i in indices.flatten():
                x,y,w,h = boxes[i]
                bbox = (x,y,w,h)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        if ok:
            cv2.putText(frame,"Tracking",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
        else:
            cv2.putText(frame,"Tracking lost",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
        tracker = cv2.TrackerKCF_create()
        ok = tracker.init(frame,bbox)
    else:
        ok,bbox = tracker.update(frame)
        x,y,w,h = bbox

        cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)
        if ok:
            cv2.putText(frame,"Tracking",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
        else:
            cv2.putText(frame,"Tracking lost",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
        
    count += 1   
    fps = cv2.getTickFrequency()/(cv2.getTickCount() - start_time)    
    cv2.putText(frame,"FPS: {:3f}".format(fps),(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
        
    cv2.imshow("Frame",frame)
    k = cv2.waitKey(25)
    if k == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()