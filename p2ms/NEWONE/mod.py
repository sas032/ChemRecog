import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pickle

labels = ["benzene","acetaminophen","acetysalicylic","adrenaline","ethane","ethene","ethylene","ibuprofen","isopentane","propylene","M-xykene (1,3 - dimethylbenzene)",\
          "o-xylene (1,2 - dimethylbenzene)","neopentane","phenylalanine","P-xylene (1,4 - dimethylbenzene)","Unknown or Bonds"]

base_dir = "data/train"
IMAGE_SIZE = 500
BATCH_SIZE = 64

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, shear_range=0.2,zoom_range=20,horizontal_flip=True,validation_split=0.1)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.2)

train_datagen=train_datagen.flow_from_directory(base_dir,target_size=(IMAGE_SIZE,IMAGE_SIZE),batch_size=BATCH_SIZE,subset='training')
test_datagen=test_datagen.flow_from_directory(base_dir,target_size=(IMAGE_SIZE,IMAGE_SIZE),batch_size=BATCH_SIZE,subset='validation')



cnn=tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64,padding='same',strides=2,kernel_size=3,activation='relu',input_shape=(500,500,3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',strides=2,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(16,activation='softmax'))


cnn.compile(optimizer=tf.keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy'])

cnn.fit(train_datagen,epochs=10,validation_data=test_datagen)

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

#pickle.dump(cnn, open('multi.pkl','wb'))
cnn.save('multi.h5')
