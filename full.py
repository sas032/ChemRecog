import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
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
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
_,contours, _ = cv2.findContours(thrash, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

cv2.imshow("img", img)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 4:
        x1 ,y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        #print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
          cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
          cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        output.write("SHAPE : Pentagon" + '\n')
    elif len(approx) == 6:
        cv2.putText(img, "Hexagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        output.write("SHAPE : Hexagon" + '\n')

    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        output.write("SHAPE : Circle" + '\n')



cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()





#for text
imgtext= Image.open('hexa.jpg')
text=tess.image_to_string(imgtext)

output.write("CHEMICAL ATTACHED :" + text + '\n')
output.close()

