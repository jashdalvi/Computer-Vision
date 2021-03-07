import numpy as np
import cv2

image = cv2.imread("../data/images/scanned-form.jpg")
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
th , image_thresh = cv2.threshold(image_gray,200,255,cv2.THRESH_BINARY)

contours , hierarchy = cv2.findContours(image_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
for i,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > max_area:
        cnt_index = i
        max_area = area

epsilon = 0.1*cv2.arcLength(contours[cnt_index],True)
approx = cv2.approxPolyDP(contours[cnt_index],epsilon,True)

width = 500
height = 700
dst_points = np.array([[width - 1,0],[0,0],[0,height - 1],[width -1 ,height -1]],dtype = np.float32).reshape(-1,1,2)

h,mask = cv2.findHomography(approx,dst_points)
document_rotated = cv2.warpPerspective(image,h,(500,700))

cv2.imshow("Document",document_rotated)

cv2.waitKey(0)

cv2.destroyAllWindows()