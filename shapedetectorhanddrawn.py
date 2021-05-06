#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 05:36:48 2021

@author: codiman
"""


import pytesseract as tess
#tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import cv2
import numpy as np
import pytesseract as tess
from PIL import Image

#output file
output=open('out.txt', 'w')



#for shape
img = cv2.imread('hexa.jpg')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
#_,contours, _ = cv2.findContours(thrash, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)






#output.write("CHEMICAL ATTACHED :" + text + '\n')
output.close()