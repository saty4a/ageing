from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

model_1 = load_model("wrinkle_detector.model")
model_2 = load_model("puffyEye_detector.model")
model_3 = load_model("darkSpot_detector.model")

def detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.05,minNeighbors=10, minSize=(60,60), flags=cv2.CASCADE_SCALE_IMAGE)
    
    faces_list = []
    preds_1 = []
    preds_2 = []
    preds_3 = []
    
    for (x,y,w,h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (120,120))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        
        faces_list.append(face_frame)
        
        if len(faces_list) > 0:
            preds_1 = model_1.predict(faces_list)
            preds_2 = model_2.predict(faces_list)
            preds_3 = model_3.predict(faces_list)
        
        for pred_1 in preds_1:
            (NoWrinkles, Wrinkled) = pred_1
            
        for pred_2 in preds_2:
            (Normal_Eyes, Puffy_Eyes) = pred_2
            
        for pred_3 in preds_3:
            (Dark_Spots, No_Spots) = pred_3
            
        label_1 = "NoWrinkles" if NoWrinkles > Wrinkled else "Wrinkled"
        color_1 = (0,255,0) if label_1 == "NoWrinkles" else (0,0,255)
        
        label_1 = "{}:{:.2f}%".format(label_1, max(NoWrinkles, Wrinkled) *100)
        
        cv2.putText(frame, label_1, (x-50, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color_1, 2)
        
        
        label_2 = "NotPuffed" if Normal_Eyes > Puffy_Eyes else "PuffyEyes"
        color_2 = (0,255,0) if label_2 == "NotPuffed" else (0,0,255)
        
        label_2 = "{}:{:.2f}%".format(label_2, max(Normal_Eyes, Puffy_Eyes) *100)
        
        cv2.putText(frame, label_2, (x-50, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_2, 2)
        
        
        label_3 = "DarkSpots" if Dark_Spots > No_Spots else "NoSpots"
        color_3 = (0,255,0) if label_3 == "NoSpots" else (0,0,255)
        
        label_3 = "{}:{:.2f}%".format(label_3, max(Dark_Spots, No_Spots) *100)
        
        
        
        cv2.putText(frame, label_3, (x-50, y-70), cv2.FONT_HERSHEY_SIMPLEX, 1, color_3, 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
        
        
        
    return frame

def render(input_file):
    img = cv2.imread(input_file)
    output = detection(img)
    cv2.imwrite("output.png", output)
    return "output"


#render("test.jpg")
