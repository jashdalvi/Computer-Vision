import cv2
import numpy as np

image = cv2.imread("cape.jpg")

cv2.imshow("image",image)

cv2.waitKey(0)

#cap = cv2.VideoCapture(0)

#while cap.isOpened():
#    ret , frame = cap.read()
'''
    if not ret:
        break

    cv2.imshow("frame",frame)
    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite('background.jpg',frame)
    elif k == ord('q'):
        break

cap.release()'''

cv2.destroyAllWindows()