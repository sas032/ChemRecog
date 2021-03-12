import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import cv2
import numpy as np
import pytesseract as tess
from PIL import Image

#output file
output=open('out.txt', 'w')
imgtext= Image.open('ocr2.png')
text=tess.image_to_string(imgtext)

output.write("CHEMICAL ATTACHED :" + text + '\n')
output.close()