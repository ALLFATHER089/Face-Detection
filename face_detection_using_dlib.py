import cv2
import dlib
image=cv2.imread(r"input_directory")
detect=dlib.get_frontal_face_detector()
fdetect=detect(image,1)
for face in fdetect:
    x,y,w,h=face.left(),face.top(),face.right(),face.bottom()
    cv2.rectangle(image,(x,y),(w,h),(0,255,255),2)
cv2.imshow("fimage",image)
cv2.waitKey(4000)
cv2.destroyAllWindows 
