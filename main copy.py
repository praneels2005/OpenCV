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

img = cv2.imread("Smiley.svg.png",1)
cv2.imwrite("smiley.png", img)
#cv2.imshow("image",img)
#print(os.getenv("API_KEY"))
genai.configure(api_key=os.getenv("API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-002")
face = PIL.Image.open("292115365.jpg")
#print(model.count_tokens(face))
#response = model.generate_content("Hello")
response = model.generate_content(["What is the emotion being portrayed by the image? Be descriptive.", face])
print(response.text)
#cv2.waitKey(0)
#cv2.destroyAllWindows()