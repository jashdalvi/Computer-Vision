import cv2
import numpy as np
import imutils

pt = None
points = []
rect = None
def onmouse(action,x,y,flags,params):
    global points,pt,rect
    if action == cv2.EVENT_LBUTTONDOWN:
        pt =(x,y)
        if pt:
            points.append(pt)
            cv2.circle(image_copy,pt,5,(0,255,0),-1)

    elif action == cv2.EVENT_LBUTTONUP:
        pt = (x,y)
        if pt:
            points.append(pt)
            cv2.circle(image_copy,pt,5,(0,255,0),-1)

        pt = None

    if len(points) == 2:
        cv2.rectangle(image_copy,points[0],points[1],(0,255,0),2)
        width = points[1][0] - points[0][0]
        height = points[1][1] - points[1][0]
        rect = (points[0][0],points[0][1],width,height)
        

image = cv2.imread("../data/images/hillary_clinton.jpg")
cv2.namedWindow("input")
cv2.setMouseCallback("input",onmouse)

image_copy =image.copy()
mask = np.zeros(image.shape[:2],dtype = np.uint8)
bgdmodel = np.zeros((1, 65), np.float64)
fgdmodel = np.zeros((1, 65), np.float64)

cv2.imshow("input",image_copy)
k = 0
while k != 27:
    cv2.imshow("input",image_copy)
    k = cv2.waitKey(1)
    if rect:
        cv2.grabCut(image,mask,rect,bgdmodel,fgdmodel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==1)|(mask==3), 255, 0).astype('uint8')
        mask2 = cv2.merge([mask2,mask2,mask2])
        output = np.uint8(image*(np.float32(mask2)/255.0))
    if k == ord('s'):
        cv2.destroyWindow("input")
        break

cv2.imshow("Output",output)

k = cv2.waitKey(0)

cv2.destroyAllWindows()