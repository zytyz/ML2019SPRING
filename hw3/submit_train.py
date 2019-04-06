import numpy as np 
import pandas as pd
import argparse 

train_X = np.load('submit_train_X.npy').reshape(-1,48,48,1)
train_Y = np.load('submit_train_Y.npy').reshape(-1,7)

print('train X shape {}'.format(train_X.shape))
print('train Y shape {}'.format(train_Y.shape))
print('\n')


ind_list = [ i for i in range(train_Y.shape[0])]
import random as rd
rd.shuffle(ind_list)
print('ind list {}'.format(ind_list[:10]))
val_len = int( 1/4 * train_Y.shape[0])

val_X = train_X[ind_list[:val_len]]
val_Y = train_Y[ind_list[:val_len]]
train_X = train_X[ind_list[val_len:]]
train_Y = train_Y[ind_list[val_len:]]

print('train X shape {}'.format(train_X.shape))
print('train Y shape {}'.format(train_Y.shape))
print('val X shape {}'.format(val_X.shape))
print('val Y shape {}'.format(val_Y.shape))


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48,48,1),data_format='channels_last' ))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(Conv2D(64, (3, 3), padding='same',data_format='channels_last' ))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))


model.add(Conv2D(128, (3, 3), padding='same',data_format='channels_last' ))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(128, (3, 3), padding='same',data_format='channels_last' ))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))


model.add(Conv2D(256, (3, 3), padding='same',data_format='channels_last' ))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(Conv2D(256, (3, 3), padding='same',data_format='channels_last' ))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.35))

model.add(Conv2D(512, (3, 3), padding='same',data_format='channels_last' ))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(Conv2D(512, (3, 3), padding='same',data_format='channels_last' ))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
callbacks = []
callbacks.append(ModelCheckpoint('submit_checkpoints/ckpt_submit.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
csv_logger = CSVLogger('submit_csv_log/log_submit.csv', separator=',', append=False)
callbacks.append(csv_logger)


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
datagen.fit(train_X)

model.fit_generator(datagen.flow(train_X, train_Y, batch_size=256),
	validation_data=(val_X, val_Y),
	steps_per_epoch= 10* len(train_X)/256 ,
	callbacks=callbacks, 
	epochs=300)

#model.fit(train_X, train_Y, epochs=200, validation_data=(val_X, val_Y), shuffle=True, batch_size=256, callbacks=callbacks)











