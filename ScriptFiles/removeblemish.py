import cv2
import numpy as np

class Blemish:
    def __init__(self,windowname,image,image_patch):
        self.windowname = windowname
        self.image = image
        self.image_patch = image_patch
        self.center = None
        self.show()
        cv2.setMouseCallback(self.windowname,self.onmouse)

    def show(self):
        cv2.imshow(self.windowname,self.image)

    def onmouse(self,action,x,y,flags,params):
        if action == cv2.EVENT_LBUTTONDOWN:
            self.center = (x,y)

        if self.center:
            mask = 255 * np.ones(self.image_patch.shape,self.image_patch.dtype)
            self.image = cv2.seamlessClone(self.image_patch,self.image,mask,self.center,cv2.NORMAL_CLONE)
            self.center = None
            self.show()


image = cv2.imread("../data/images/blemish.png")
image_patch = image.copy()[90:120,200:230]

blemish = Blemish("image",image,image_patch)

while True:
    k = cv2.waitKey(25)
    if k == 27:
        break
    cv2.imshow(blemish.windowname,blemish.image)

cv2.destroyAllWindows()