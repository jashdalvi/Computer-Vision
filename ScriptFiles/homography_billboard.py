import cv2
import numpy as np

class FourPoints():
    def __init__(self,window_name,src_image,src_points,times_square_image):
        self.dst_points = []
        self.window_name = window_name
        self.src_image = src_image
        self.src_image_copy = src_image.copy()
        self.point = None
        self.src_points = src_points
        self.times_square_image = times_square_image
        self.times_square_image_copy = times_square_image.copy()
        self.show()
        cv2.setMouseCallback(self.window_name,self.onmouse)


    def show(self):
        cv2.imshow(self.window_name,self.times_square_image)

    def onmouse(self,action,x,y,flags,params):
        if action == cv2.EVENT_LBUTTONDOWN:
            self.point = [x,y]
            self.dst_points.append(self.point)

        elif action == cv2.EVENT_LBUTTONUP:
            self.point = None
        
        if self.point:
            cv2.circle(self.times_square_image,(self.point[0],self.point[1]),5,(0,255,0),-1,cv2.LINE_AA)
            self.show()

        if len(self.dst_points) == 4:
            dst_points = np.float32(self.dst_points)
            h,_ = cv2.findHomography(self.src_points,dst_points)
            output_dim = (self.times_square_image.shape[1],self.times_square_image.shape[0])
            output_image = cv2.warpPerspective(self.src_image_copy,h,output_dim)
            output_image_gray = cv2.cvtColor(output_image,cv2.COLOR_BGR2GRAY)
            th,mask = cv2.threshold(output_image_gray,50,255,cv2.THRESH_BINARY)
            mask_3d = cv2.merge([mask,mask,mask])
            mask = np.float32(mask_3d)/255.0
            times_square = cv2.multiply(self.times_square_image_copy, 1 - mask_3d)
            final_output = cv2.add(np.uint8(times_square),np.uint8(output_image))
            cv2.imshow("Homography image",final_output)

    @staticmethod
    def destroy():
        cv2.destroyWindow("Homography image")


src_image = cv2.imread("../data/images/first-image.jpg")
times_square_image = cv2.imread("../data/images/times-square.jpg")
times_square_image = cv2.resize(times_square_image,(500,500),interpolation = cv2.INTER_CUBIC)
window_name = "Billboard image"

(H,W) = src_image.shape[:2]
src_points = np.float32([[0,0],[int(W-1),0],[int(W-1),int(H-1)],[0,int(H-1)]])
four_points = FourPoints(window_name,src_image,src_points,times_square_image)

k = 0 
while k != 27:
    k = cv2.waitKey(25) & 0xFF
    if k == ord('q'):
        four_points.destroy()
        break
    

cv2.destroyAllWindows()

