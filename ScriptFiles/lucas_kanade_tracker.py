import cv2
import numpy as np


video_path = "../data/videos/cycle.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('../data/videos/cycle_tracker.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),20,(width,height))

ret,old_frame = cap.read()
old_frame_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)

old_points = cv2.goodFeaturesToTrack(old_frame_gray,100,0.3,7)
mask = np.zeros_like(old_frame,dtype = np.uint8)
while True:

    ret ,new_frame = cap.read()

    if not ret:
        break
    
    new_frame_gray = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)

    new_points , status, _ = cv2.calcOpticalFlowPyrLK(old_frame_gray,new_frame_gray,old_points,None)

    good_old_points = old_points[status == 1]
    good_new_points = new_points[status == 1]

    for i,(old,new) in enumerate(zip(good_old_points,good_new_points)):

        a,b = old.ravel()
        c,d = new.ravel()
        cv2.line(mask,(a,b),(c,d),(0,255,0),2,cv2.LINE_AA)
        cv2.circle(new_frame,(a,b),2,(0,255,0),-1)

    save_frame = cv2.add(new_frame,mask)

    out.write(save_frame)

    cv2.imshow("Optical Flow",save_frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    old_points = new_points
    old_frame_gray = new_frame_gray


cap.release()
out.release()
cv2.destroyAllWindows()
