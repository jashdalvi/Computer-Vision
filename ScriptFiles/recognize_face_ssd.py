import cv2
import numpy as np
import imutils

weights_file = "../data/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "../data/models/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(config_file,weights_file)



cap = cv2.VideoCapture(0)


while cap.isOpened():

    ret , frame = cap.read()

    start_time = cv2.getTickCount()

    if not ret:
        break

    frame = imutils.resize(frame,height = 500)

    height_frame , width_frame = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)

    output = net.forward()

    for i in range(output.shape[2]):
        score = float(output[0,0,i,2])

        x1 = int(output[0,0,i,3]*width_frame)
        y1 = int(output[0,0,i,4]*height_frame)
        x2 = int(output[0,0,i,5]*width_frame)
        y2 = int(output[0,0,i,6]*height_frame)

        if score > 0.5:

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - start_time)    
    cv2.putText(frame,"FPS: {:.3f}".format(fps),(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("Face Detection",frame)

    k = cv2.waitKey(25)
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



