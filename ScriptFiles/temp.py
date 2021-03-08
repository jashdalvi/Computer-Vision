import cv2
import numpy as np

image = cv2.imread("shirt.jpg")

cv2.imshow("image",image)

cv2.waitKey(0)

cv2.destroyAllWindows()