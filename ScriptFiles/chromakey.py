import cv2
import numpy as np

center = None
value = np.zeros((3,),dtype = np.uint8)
lower_green = np.array([50,100,100],dtype = np.uint8)
upper_green = np.array([60,255,255],dtype = np.uint8)
tolerance_value = 1

binary_mask = None
def onmouse(action,x,y,flags,params):
    global center,value,binary_mask,lower_green,upper_green
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x,y)
    elif action == cv2.EVENT_LBUTTONUP:
        center = None

    if center:
        frame_hsv_patch = frame_hsv[y-10:y+10,x-10:x+10]
        for i in range(3):
            value[i] = frame_hsv_patch[:,:,i].mean().astype("uint8")
        lower_green = np.array([value[0]-5,100,100],dtype = np.uint8)
        upper_green = np.array([value[0]+5,255,255],dtype = np.uint8)
        binary_mask = cv2.inRange(frame_hsv,lower_green,upper_green)

        binary_mask = cv2.merge([binary_mask,binary_mask,binary_mask])
        binary_mask = np.float32(binary_mask)/255
        inter = cv2.multiply(np.float32(frame)/255,1-binary_mask)
        cv2.imshow("Frame",inter)
    if not np.any(binary_mask):
        cv2.imshow("Frame",frame)



def tolerance(*args):
    global tolerance_value,lower_green,upper_green,value,center,binary_mask
    tolerance_value = args[0]
    lower_green = np.array([value[0]-5,100,100],dtype = np.uint8)
    upper_green = np.array([value[0]+5,255,255],dtype = np.uint8)
    lower_green = cv2.subtract(lower_green,np.uint8([tolerance_value,tolerance_value,tolerance_value]))
    upper_green =cv2.add(upper_green,np.uint8([tolerance_value,tolerance_value,tolerance_value]))

    binary_mask = cv2.inRange(frame_hsv,lower_green,upper_green)

    binary_mask = cv2.merge([binary_mask,binary_mask,binary_mask])
    binary_mask = np.float32(binary_mask)/255
    inter = cv2.multiply(np.float32(frame)/255,1-binary_mask)
    cv2.imshow("Frame",inter)



cap = cv2.VideoCapture("greenscreen-asteroid.mp4")
bg = cv2.imread("image1.jpg")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",onmouse)
cv2.createTrackbar("Tolerance-slider","Frame",tolerance_value,100,tolerance)

ret,frame = cap.read()
frame = cv2.resize(frame,(400,400),interpolation=cv2.INTER_CUBIC)
bg  = cv2.resize(bg,(400,400),interpolation=cv2.INTER_CUBIC)
cv2.putText(frame,"First select the patch then slide the trackbar and press s",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
cv2.imshow("Frame",frame)
frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
k = cv2.waitKey(0)
i = 0
if k == ord('s'):
    while cap.isOpened():
        ret , frame1 = cap.read()
        if i == 0:
            cv2.destroyWindow("Frame")
            i +=1
        if not ret:
            break

        frame1 = cv2.resize(frame1,(400,400),interpolation=cv2.INTER_CUBIC)
        bg  = cv2.resize(bg,(400,400),interpolation=cv2.INTER_CUBIC)

        frame_hsv = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)

        binary_mask = cv2.inRange(frame_hsv,lower_green,upper_green)

        binary_mask = cv2.merge([binary_mask,binary_mask,binary_mask])
        binary_mask = np.float32(binary_mask)/255
        inter = cv2.multiply(np.float32(frame1)/255,1-binary_mask)
        inter1 = cv2.multiply(np.float32(bg)/255,binary_mask)
        output = cv2.add(inter,inter1)

        output = (output*255).astype("uint8")
        binary_mask = (binary_mask*255).astype("uint8")
        cv2.imshow("Frame1",output)

        k = cv2.waitKey(50)
        if k == ord('q'):
            break

cv2.destroyAllWindows()