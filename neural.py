import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, COnv2D, MaxpPooling2d, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
plt.style.use('fivethirtyeight')



from keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data() #all data must be in np array

print(x_train.shape)  #first = no of data | sec and third = rows containing that by that images | fourth = depth of images
print(y_train.shape)  #first data | two column
print(x_test.shape)
print(y_test.shape)



index = 0
print(x_train[index]) #see image as an array

img = plt.imshow(x_train[index]) #show as image


print('Label of Image : ', y_train[index])


#neural
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#normalised
x_train = x_train/255
x_test = x_test/255

model = Sequential()
model.add(Conv2D(32,(5,5),activation='relu',imput_shape(32,32,3)))
model.add(MaxPolling2d(pool_size=(2,2)))
model.add(Conv2D(32,(5,5),activation='relu',))
model.add(MaxPolling2d(pool_size=(2,2)))
model.add(Flatten())
 
 
#1000 neurons
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(250,activation='relu')
model.add(Dense(10,activation='softmax')

mdoel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#training
hist=model.fit(x_train,y_train_one_hot. batch_size = 256, epochs = 10,validation_split = 0.2)


#test
model.evaluate(x_test,y_test_one_hot)[1]




#visualising epoch ~ accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()




image = file.upload()

res_img=resize(img,(32,32,3)
pred = model.predict(np.array([res_image]))
print(pred)
list_i=[0,1,2,3,4,5,6,7,8,9]
for i in range(10):
	for j in range(10):
		if x[0][list_i[i]> x[0][list_i[j]:
			temp=list_i[i]
			list_i[i]=list_i[j] 
			list_i[j]=temp
			
			
print(list_i[0])
