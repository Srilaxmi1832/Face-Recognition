import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('C:/Users/manik/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w]
    return cropped_face



cap=cv2.VideoCapture(1)
count=0
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(400,400))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_name_Path = "C:/Users/manik/Desktop/bro files/faces/user" + str(count) + '.jpg'
        cv2.imwrite(file_name_Path,face)

        cv2.putText(face,str(count),(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)

    else:
        print('Face Not Found')
        pass
    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()


