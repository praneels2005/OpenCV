import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
import cv2
import PIL.Image
import pyautogui
import imutils
load_dotenv()


genai.configure(api_key=os.getenv("API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-002")
face = PIL.Image.open("smiley.png")
    #print(model.count_tokens(face))
    #response = model.generate_content("Hello")

response = model.generate_content(["You are a face coach, describe what you see inside the shapes on the person's face and give a detailed breakdown of these features. Give me a rating out of 10 for this person.", face],stream=True)
for chunk in response:
    print(chunk.text,end=" ")