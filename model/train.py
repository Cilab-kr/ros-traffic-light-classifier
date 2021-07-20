#!/usr/bin/env python

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model
from tensorflow.keras.layers import Activation, Dropout, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import ModelCheckpoint

# In[2]:


import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[3]:


batch_size = 64
nb_epoch = 100 
nb_classes = 3
image_shape = [64, 64, 3]


# In[4]:


model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same', input_shape=image_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# In[5]:


model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()


# In[100]:


datagen = ImageDataGenerator(width_shift_range=.2, height_shift_range=.2, shear_range=0.05, zoom_range=.1,
                             fill_mode='nearest', rescale=1. / 255)
image_data_gen = datagen.flow_from_directory('images', target_size=(64, 64), classes=['green', 'red', 'unknown'],
                                             batch_size=batch_size)


# In[101]:
checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', verbose=1,
        save_best_only=True, mode='auto', period=1)

model.fit_generator(generator=image_data_gen,
                    epochs=nb_epoch,
                    use_multiprocessing=True,
                    workers=4,
                    callbacks=[checkpoint])


# In[102]:


model.save('light_classifier_model.h5')


# In[6]:


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[ ]:




