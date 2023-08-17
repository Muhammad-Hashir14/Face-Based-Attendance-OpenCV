import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_data = []
name = input("Enter Your Name : ")
i = 0
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3, 5)
    for (x,y,w,h) in faces:
        crop_image = frame[y:y+h, x:x+w,:]
        resized_image = cv2.resize(crop_image,(50,50))
        if (len(face_data) <= 100) and (i%10 == 0):
            face_data.append(resized_image)
        i += 1
        cv2.putText(frame, str(len(face_data)), (50,50),cv2.FONT_HERSHEY_COMPLEX,1, (50,50,255),1) 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,128,0),1)
    cv2.imshow('frame',frame)
    k= cv2.waitKey(1)
    if k==ord('q') or len(face_data) == 100:
        break
print(face_data)
video.release()
cv2.destroyAllWindows()

face_data = np.asarray(face_data)
face_data = face_data.reshape(100,-1)

if "data.pkl" not in os.listdir("data/"):
    names = [name] * 100
    with open('data/data.pkl','wb') as f:
        pickle.dump(names,f)
else:
    with open("data/data.pkl",'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open('data/data.pkl','wb') as f:
        pickle.dump(names,f)

if "face_data.pkl" not in os.listdir("data/"):
    with open('data/face_data.pkl','wb') as f:
        pickle.dump(face_data,f)
else:
    with open("data/face_data.pkl",'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis = 0)   
    with open('data/face_data.pkl','wb') as f:
        pickle.dump(faces,f)
    
print(face_data)
