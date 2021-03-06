import cv2
import numpy as np
import imutils


weights_file ="../data/models/pose_iter_160000.caffemodel"
config_file = "../data/models/mpi.prototxt"

net = cv2.dnn.readNetFromCaffe(config_file,weights_file)

pose_pairs = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()
    start_time = cv2.getTickCount()

    if not ret:
        break

    frame = imutils.resize(frame,height = 800)

    frame_height,frame_width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255,(368,368), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(blob)

    output = net.forward()

    points = []

    for i in range(output.shape[1]):

        prob_map = output[0,i,:,:]

        prob_map_height, prob_map_width = prob_map.shape[:2]

        minval,prob,minpoint,point = cv2.minMaxLoc(prob_map)

        x = int((point[0]/prob_map_width) * frame_width)
        y = int((point[1]/prob_map_height)*frame_height)

        if prob > 0.1:
            points.append((x,y))

        else:
            points.append(None)

        
    for pair in pose_pairs:
        point1 = pair[0]
        point2 = pair[1]

        if points[point1] and points[point2]:
            cv2.circle(frame,points[point1],5,(0,0,255),-1,cv2.LINE_AA)
            cv2.circle(frame,points[point2],5,(0,0,255),-1,cv2.LINE_AA)
            cv2.line(frame,points[point1],points[point2],(255,0,0),2,cv2.LINE_AA)

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - start_time)
    cv2.putText(frame,"FPS: {:.3f}".format(fps),(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("Pose Estimation",frame)
    k = cv2.waitKey(25)
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()