import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
import cv2
import PIL.Image
import pyautogui
import imutils
load_dotenv()

#Locating coordinates on the screen Command+Shift+4
#Contouring

cap = cv2.VideoCapture(0)
Face_path = "haarcascade_frontalface_default.xml"
Eye_path = "haarcascade_eye.xml"
Mouth_path = "mouth.xml"
Nose_path = "nose.xml"

face_cascade = cv2.CascadeClassifier(Face_path)
eyes_cascade = cv2.CascadeClassifier(Eye_path)
mouth_cascade = cv2.CascadeClassifier(Mouth_path)
nose_cascade = cv2.CascadeClassifier(Nose_path)

#lower "scaleFactor" can improve accuracy but slow down detection
#higher "minNeighbors" reduces false positives but might miss some objects

while(True):
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (0,0),fx=0.5, fy=0.5 )
    
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.10,minNeighbors=5,minSize=(40,40))
    eyes = eyes_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.02,minNeighbors=20,minSize=(10,10))
    Mouth = mouth_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.10,minNeighbors=20,minSize=(30,30))
    Nose = nose_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.02,minNeighbors=15,minSize=(10,10))
    eyes_array = np.array(eyes)
    ''' for (i) in range(len(eyes_array)-1):
		print(eyes_array[i],"eye")
		print(eyes_array[i+1],"eye2")
		#print(j, "eye2")'''
    for (x,y,w,h),(i),(mx, my, mw, mh), (nx, ny, nw, nh) in zip(faces,range(len(eyes_array)-1), Mouth,Nose):
        #eye1
        xc1 = (eyes_array[i][0]+ (eyes_array[i][0]+eyes_array[i][2]))/2
        yc1 = (eyes_array[i][1]+ (eyes_array[i][1]+eyes_array[i][3]))/2
        radius1 = eyes_array[i][2]/2
    
        #eye2
        xc2 = (eyes_array[i+1][0]+ (eyes_array[i+1][0]+eyes_array[i+1][2]))/2
        yc2 = (eyes_array[i+1][1]+ (eyes_array[i+1][1]+eyes_array[i+1][3]))/2
        radius2 = eyes_array[i+1][2]/2
        
        #Mouth
        mcx = (mx+(mx+mw))/2
        mcy = (my+(my+mh))/2
        mouth_radius = mw/2
        cv2.circle(frame, (int(mcx), int(mcy)), int(mouth_radius), (255,0,0),2)
        
        #Nose
        cv2.rectangle(frame, (nx, ny),(nx+nw, ny+nh), (255,0,0),2)
        
        #Face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        if(len(eyes_array) == 2):
            cv2.circle(frame, (int(xc1), int(yc1)), int(radius1), (255,0,0),2)
            cv2.circle(frame, (int(xc2), int(yc2)), int(radius2), (255,0,0),2)
    
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # Set the window to fullscreen
    cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Camera", frame)
    
    ch = cv2.waitKey(1)
    if ch == ord('q'):
        break
    elif ch == ord('s'):
        ss = pyautogui.screenshot()
        #ss = pyautogui.screenshot(region=[0,66, 962, 539])
        ss = cv2.cvtColor(np.array(ss), cv2.COLOR_RGB2BGR)
        cv2.imwrite("screenshot.png", ss)


img = cv2.imread("292115365.jpg", 0)
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

index = -1
thickness = 4
color = (255, 0 ,255)

cv2.drawContours(img, )



#genai.configure(api_key=os.getenv("API_KEY"))
#model = genai.GenerativeModel("gemini-1.5-flash-002")
#face = PIL.Image.open("screenshot.png")
    #print(model.count_tokens(face))
    #response = model.generate_content("Hello")

#response = model.generate_content(["What do you see in the image? Be descriptive.", face],stream=True)
#for chunk in response:
#    print(chunk.text,end=" ")
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()