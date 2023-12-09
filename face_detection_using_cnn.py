import cv2
import dlib

# Load the CNN face detection model
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Load the image
image = cv2.imread(r"input directory")
image=cv2.resize(image,(800,600))

# Perform face detection using the detector object
fdetect = detector(image,1)

for face in fdetect:
    x, y, x2, y2 = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
    print(face.confidence)
    cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 255), 2)

cv2.imshow("fimage", image)
cv2.waitKey(4000)
cv2.destroyAllWindows()
cv2.destroyAllWindows 