import numpy as np
import keras
import h5py
from keras.models import load_model
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) =mnist.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test,num_classes=10)
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
print(y_train.shape)
model = Sequential()
model.add(Convolution2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
model.add(Convolution2D(32,3,padding='same',data_format="channels_last",activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
model.add(Convolution2D(64,(3,3),padding='same',activation='relu'))
model.add(Convolution2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(20,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="sigmoid"))
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=1,batch_size=254)
score = model.evaluate(x_test,y_test, batch_size=254)
model.save('digits.h5')
result=model.predict(x_test);
print(result)
print("moel accuray is:-",score)
