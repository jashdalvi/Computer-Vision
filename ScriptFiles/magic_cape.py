import numpy as np
import cv2

cap = cv2.VideoCapture(0)

bg = cv2.imread("background.jpg")

width = int(cap.get(3))
height = int(cap.get(4))
out = cv2.VideoWriter("magic.mp4",cv2.VideoWriter_fourcc(*"MJPG"),25,(width,height))
lower_blue = np.array([100,100,0],dtype = np.uint8)
upper_blue = np.array([255,255,50],dtype = np.uint8)



while cap.isOpened():

    ret,frame = cap.read()
    if not ret:
        break
    

    mask = cv2.inRange(frame, lower_blue, upper_blue)
    mask = cv2.merge([mask,mask,mask])
    mask = np.float32(mask)/255.0

    inter1 = (frame * (1 - mask)).astype("uint8")
    inter2 = (bg * mask).astype("uint8")
    output = cv2.add(inter1,inter2)

    cv2.imshow("Frame",output)
    out.write(output)
    k = cv2.waitKey(25)
    if k == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
