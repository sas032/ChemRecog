import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

labels = ["benzene","acetaminophen","acetysalicylic","adrenaline","ethane","ethene","ethylene","ibuprofen","isopentane","propylene","M-xykene (1,3 - dimethylbenzene)",\
          "o-xylene (1,2 - dimethylbenzene)","neopentane","phenylalanine","P-xylene (1,4 - dimethylbenzene)","Unknown or Bonds"]

image_height = 500
image_width = 500
batch_size = 2

model = keras.Sequential([
    layers.Input((500,500,1)),
    layers.Conv2D(16,3,padding = 'same'),
    layers.Conv2D(32,3,padding = 'same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    #layers.Dense(10)
    layers.Dense(16,activation='softmax')
])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'Datasets/',
    labels='inferred',
    label_mode = "int",
    #color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_height,image_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training"
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'Datasets/',
    labels='inferred',
    label_mode = "int",
    #color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_height,image_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation"
)

def augment(x,y):
    image = tf.image.random_brightness(x, max_delta = 0.5)
    return image, y

ds_train = ds_train.map(augment)

# Custom Loop
for epochs in range(10):
    for x,y in ds_train:
        #train here
        pass

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss=[
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ],
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=2, verbose=2)

#test
img = cv2.imread("Datasets/Ethene/Ethene.jpeg")
res=cv2.resize(img ,dsize=(500,500), interpolation=cv2.INTER_CUBIC)
npimg=np.array(res)
print(npimg.shape)
#npimg.shape=(500,500,1)
sw=np.moveaxis(npimg,0,0)
rr=np.expand_dims(sw,0)
prediction = model.predict(npimg)
pred_list=list(prediction[0])
index = pred_list.index(max(pred_list))
print(labels[index])