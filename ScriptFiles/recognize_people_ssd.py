import cv2
import numpy as np
import imutils

weights_file = "../data/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
config_file = "../data/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
class_file = "../data/models/coco_class_labels.txt"

net = cv2.dnn.readNetFromTensorflow(weights_file,config_file)

with open(class_file,'r') as f:
    classes = f.read().rstrip("\n").split("\n")

cap = cv2.VideoCapture("../data/videos/pedestrians.mp4")

person_id = classes.index("person")

while cap.isOpened():

    ret , frame = cap.read()

    start_time = cv2.getTickCount()

    if not ret:
        break

    frame = imutils.resize(frame,height = 500)

    height_frame , width_frame = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0/127.5, (300, 300),[127.5,127.5,127.5] ,True)

    net.setInput(blob)

    output = net.forward()

    for i in range(output.shape[2]):

        class_id = int(output[0,0,i,1])
        score = float(output[0,0,i,2])

        x1 = int(output[0,0,i,3]*width_frame)
        y1 = int(output[0,0,i,4]*height_frame)
        x2 = int(output[0,0,i,5]*width_frame)
        y2 = int(output[0,0,i,6]*height_frame)

        if class_id == person_id and score > 0.5:

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - start_time)    
    cv2.putText(frame,"FPS: {:3f}".format(fps),(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("Pedestrain Detection",frame)

    k = cv2.waitKey(25)
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



