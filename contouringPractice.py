import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
import cv2
import PIL.Image
import pyautogui
import imutils

img = cv2.imread("292115365.jpg", 1)
cv2.imshow("Original", img)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = img.copy()
index = -1
thickness = 4
color = (255, 0 ,0)

cv2.drawContours(img2, contours, index, color, thickness)

cv2.imshow("Contours", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()