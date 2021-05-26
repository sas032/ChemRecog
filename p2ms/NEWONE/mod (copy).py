import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pickle

from tensorflow import keras

labels = ["benzene","acetaminophen","acetysalicylic","adrenaline","ethane","ethene","ethylene","ibuprofen","isopentane","propylene","M-xykene (1,3 - dimethylbenzene)",\
          "o-xylene (1,2 - dimethylbenzene)","neopentane","phenylalanine","P-xylene (1,4 - dimethylbenzene)","Unknown or Bonds"]


cnn = keras.models.load_model('multi.h5')

img = cv2.imread("data/test/")
res=cv2.resize(img ,dsize=(500,500), interpolation=cv2.INTER_CUBIC)
npimg=np.array(res)
print(npimg.shape)
sw=np.moveaxis(npimg,0,0)
rr=np.expand_dims(sw,0)
prediction = cnn.predict(rr)
pred_list=list(prediction[0])
index = pred_list.index(max(pred_list))
print(labels[index])


