import cv2
import numpy as np

class Sketcher:
    def __init__(self,windowname,dests,colors):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors = colors
        self.show()
        cv2.setMouseCallback(self.windowname,self.onmouse)

    def show(self):
        cv2.imshow(self.windowname,self.dests[0])
        cv2.imshow(self.windowname + "-mask",self.dests[1])

    def onmouse(self,action,x,y,flags,params):
        pt = (x,y)
        if action == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif action == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None
        
        if self.prev_pt :
            for dst,color in zip(self.dests,self.colors):
                cv2.line(dst,self.prev_pt,pt,color,5)
            self.prev_pt = pt
            self.show()

image = cv2.imread("../data/images/Lincoln.jpg")

image_mask = image.copy()
mask = np.zeros(image_mask.shape[:2],dtype= np.uint8)

sketch = Sketcher("image",[image_mask,mask],[(0,255,0),255])

k = 0

while k!=27:
    k = cv2.waitKey(25) & 0xFF
    if k == ord('t'):
        output_telea = cv2.inpaint(image_mask,mask,3,cv2.INPAINT_TELEA)
        cv2.imshow("TELEA",output_telea)
        #k = cv2.waitKey(0)
        #if k == 27:
        #    break
    if k == ord('n'):
        output_ns = cv2.inpaint(image_mask,mask,3,cv2.INPAINT_NS)
        cv2.imshow("Navier stokes",output_ns)
        #k = cv2.waitKey(0)
        #if k == 27:
        #    break
    if k == ord('r'):
        image_mask[:] = image.copy()
        mask [:,:] = 0
        sketch.show()

cv2.destroyAllWindows()

