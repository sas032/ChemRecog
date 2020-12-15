import pytesseract as tess
from PIL import Image
img= Image.open('ocr2.png')

text=tess.image_to_string(img)

print(text)