import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('one.jpeg')

dst = cv.fastNlMeansDenoisingColored(img,None,10,10,10,100)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
#plt.show()
plt.imsave('dOne.png', dst, cmap='gray', vmin=0, vmax=255)