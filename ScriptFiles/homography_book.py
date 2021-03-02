import cv2
import numpy as np

class FourPoints():
    def __init__(self,window_name,src_image,dst_points):
        self.src_points = []
        self.window_name = window_name
        self.src_image = src_image
        self.point = None
        self.dst_points = dst_points
        self.show()
        cv2.setMouseCallback(self.window_name,self.onmouse)


    def show(self):
        cv2.imshow(self.window_name,self.src_image)

    def onmouse(self,action,x,y,flags,params):
        if action == cv2.EVENT_LBUTTONDOWN:
            self.point = [x,y]
            self.src_points.append(self.point)

        elif action == cv2.EVENT_LBUTTONUP:
            self.point = None
        
        if self.point:
            cv2.circle(self.src_image,(self.point[0],self.point[1]),5,(0,255,0),-1,cv2.LINE_AA)
            self.show()

        if len(self.src_points) == 4:
            src_points = np.float32(self.src_points)
            h,_ = cv2.findHomography(src_points,self.dst_points)
            output_image = cv2.warpPerspective(self.src_image,h,(300,400))
            cv2.imshow("Homography image",output_image)

    @staticmethod
    def destroy():
        cv2.destroyWindow("Homography image")


src_image = cv2.imread("../data/images/book1.jpg")
window_name = "Book image"

src_image = cv2.resize(src_image,None,fx = 1,fy = 0.75,interpolation=cv2.INTER_CUBIC)

dst_points = np.float32([[0,0],[299,0],[299,399],[0,399]])
four_points = FourPoints(window_name,src_image,dst_points)

k = 0 
while k != 27:
    k = cv2.waitKey(25) & 0xFF
    if k == ord('q'):
        four_points.destroy()
        break
    

cv2.destroyAllWindows()

