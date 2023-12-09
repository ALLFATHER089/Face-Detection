import cv2
path=r'input_directory'
image = cv2.imread(path)


image=cv2.resize(image,(800,600))
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


Edetect=cv2.CascadeClassifier('directory/haarcascade_eye.xml')
fdetect=cv2.CascadeClassifier('directory/haarcascade_frontalface_default.xml')

Eye_detect = Edetect.detectMultiScale(image_gray,scaleFactor=1.16,minNeighbors=9,minSize=(20,20), maxSize=(45,45))
face_detect = fdetect.detectMultiScale(image_gray, scaleFactor = 1.3, minSize = (30,30))
print(face_detect)
len(face_detect)

for (x,y,w,h) in Eye_detect:
  print(w,h)
  
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 1)


for (x,y,w,h) in face_detect:
  
  
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 3)

cv2.imshow("Fimage",image)
cv2.waitKey(4000)
cv2.destroyAllWindows 