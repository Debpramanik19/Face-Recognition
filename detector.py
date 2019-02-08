import cv2
import numpy as np

from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceDetect = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.5,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="Deb"
        elif(id==2):
            id="kutus"
        else:
            id="Unknown"
        cv2.putText(im,str(id),(x,y+h),font,color=(255,0,0),fontScale=3)
    cv2.imshow("Face",im) 
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
