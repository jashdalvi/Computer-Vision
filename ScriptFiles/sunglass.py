import cv2
import numpy as np

def aplha_blend(face_roi,sunglass):
    face_roi_height = face_roi.shape[0]
    eye_roi = face_roi[int(face_roi_height/4):int(face_roi_height/2),:]
    sunglass_resize = cv2.resize(sunglass,(eye_roi.shape[1],eye_roi.shape[0]),interpolation = cv2.INTER_CUBIC)
    sunglass_mask = sunglass_resize[:,:,3]
    sunglass_image = sunglass_resize[:,:,:3]
    sunglass_mask = np.float32(sunglass_mask)/255
    sunglass_mask = cv2.merge([sunglass_mask,sunglass_mask,sunglass_mask])
    eye_glass = (eye_roi *(1 - sunglass_mask)).astype("uint8")
    sunglass_inter = (sunglass_image * sunglass_mask).astype("uint8")
    output = cv2.add(eye_glass,sunglass_inter)
    output_weighted = cv2.addWeighted(eye_roi,0.4,output,0.6,0)
    face_roi[int(face_roi_height/4):int(face_roi_height/2),:] = output_weighted

    return face_roi


weights_file = "../data/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "../data/models/deploy.prototxt"

sunglass = cv2.imread("../data/images/sunglass.png",-1)

net = cv2.dnn.readNetFromCaffe(config_file,weights_file)
cap =cv2.VideoCapture(0)

while cap.isOpened():

    ret,frame = cap.read()

    start_time = cv2.getTickCount()

    if not ret:
        break


    height_frame , width_frame = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)

    output = net.forward()

    for i in range(output.shape[2]):
        score = float(output[0,0,i,2])

        x1 = int(output[0,0,i,3]*width_frame)
        y1 = int(output[0,0,i,4]*height_frame)
        x2 = int(output[0,0,i,5]*width_frame)
        y2 = int(output[0,0,i,6]*height_frame)

        if score > 0.5:
            
            face_roi = frame.copy()[y1:y2,x1:x2]
            face_roi = aplha_blend(face_roi,sunglass)
            frame[y1:y2,x1:x2] = face_roi

    fps = cv2.getTickFrequency()/(cv2.getTickCount() - start_time)    
    cv2.putText(frame,"FPS: {:.3f}".format(fps),(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("Face Detection",frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()