from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, DepthwiseConv2D,BatchNormalization, LeakyReLU, AveragePooling2D,GlobalAvgPool2D,Dropout
import numpy as np 
import pandas as pd 
import argparse
from multiprocessing import Pool
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam


def processcmd(): 
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-trainpath',default='data/train.csv',type=str)
    parser.add_argument('-num','--modelnum',default=0,type=int)
    parser.add_argument('-reload',default=-1,type=int)
    #parser.add_argument('-batch_size',default=256,type=int)
    #parser.add_argument('-type','--modeltype',default='mobileNet',type=str)
    parser.add_argument('-epochs',default=200,type=int)
    parser.add_argument('-lr',default=0.001,type=float)
    parser.add_argument('-drop',action='store_true')
    #parser.add_argument('-half',default=False,type=bool)
    args = parser.parse_args()
    print(args)
    return args


def getimgs(raw):
    img = np.array(raw[1].split(' ')).reshape(48,48,1)
    return img

def getlabels(raw):
    label = np.zeros(7)
    idx = int(raw[0])
    label[idx] = 1
    return label

def mydatasplit(x_total,y_total):
    assert len(x_total)==len(y_total)
    if args.reload!=-1:
        with open('val_ind/ind_list'+str(args.reload)+'.txt') as f:
            ind_list = f.read().split(' ')
            ind_list = [int(x) for x in ind_list]
    else:
        try:
            with open('val_ind/ind_list'+str(args.modelnum)+'.txt') as f:
                ind_list = f.read().split(' ')
                ind_list = [int(x) for x in ind_list]
        except:
            ind_list = [ i for i in range(x_total.shape[0])]
            import random as rd
            rd.shuffle(ind_list)
    print('ind list {}'.format(ind_list[:10]))
    val_len = int( 1/5 * x_total.shape[0])

    path = 'val_ind/ind_list'+str(args.modelnum)+'.txt'
    with open(path,'w') as f:
        print('writing ind list in '+path)
        f.write(' '.join([str(x) for x in ind_list]))
    #print('total data length {}'.format(totaldata.shape[0]))
    x_val = x_total[ind_list[:val_len]]
    y_val = y_total[ind_list[:val_len]]
    x_train = x_total[ind_list[val_len:]]
    y_train = y_total[ind_list[val_len:]]
    return x_train,y_train,x_val,y_val

def readfile(path):
    print("Reading File...")
    raw_train = pd.read_csv(path).values
    print('raw train shape {}'.format(raw_train.shape))

    p = Pool()
    x_train_total = p.map(getimgs,raw_train)
    x_train_total = np.array(x_train_total,dtype=float)
    x_train_total /= 255.0

    #y_train_total = np.array(raw_train[:,0],dtype=int).reshape(-1,1)
    y_train_total = p.map(getlabels,raw_train)
    y_train_total = np.array(y_train_total).astype(np.int)
    print(y_train_total.shape)
    print(x_train_total.shape)
    print(y_train_total.shape[:5])
    print(x_train_total.shape[0])

    x_train,y_train,x_val,y_val = mydatasplit(x_train_total,y_train_total)
    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_val', x_val.shape)
    print('y_val', y_val.shape)

    return x_train,y_train,x_val,y_val

def conv_block(inp,filters,strides):
    x = Conv2D(filters,kernel_size=(3,3),strides=strides,padding='same',use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    return x

def dw_block(inp,filters,strides):
    x = DepthwiseConv2D(kernel_size=(3,3),strides=strides,padding='same',use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2D(filters,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    return x


args = processcmd()
x_train,y_train,x_val,y_val = readfile(args.trainpath)    # 'train.csv'

scale=1

img = Input(shape=(48,48,1))
x = conv_block(img,int(64*scale),(2,2))
x = dw_block(x,int(64*scale),(1,1))
x = dw_block(x,int(200*scale),(2,2))
x = dw_block(x,int(128*scale),(1,1))
x = dw_block(x,int(128*scale),(1,1))
x = dw_block(x,int(128*scale),(1,1))
x = dw_block(x,int(128*scale),(1,1))
x = GlobalAvgPool2D()(x)
if args.drop:
    x = Dropout(0.5)(x)
x = Dense(7,activation='softmax')(x)

model = Model(inputs=img,outputs=x)
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


callbacks = []
callbacks.append(ModelCheckpoint('ckpt/ckpt'+str(args.modelnum)+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
csv_logger = CSVLogger('log/log'+str(args.modelnum)+'.csv', separator=',', append=False)
callbacks.append(csv_logger)
earlystop = EarlyStopping(monitor='val_acc', patience=20 ,mode='max')
callbacks.append(earlystop)

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
datagen.fit(x_train)

print('start training...')

model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
    validation_data=(x_val, y_val),
    steps_per_epoch= 10* len(x_train)/256 ,
    callbacks=callbacks, 
    epochs=args.epochs)

#model.fit(train_X, train_Y, epochs=200, validation_data=(val_X, val_Y), shuffle=True, batch_size=256, callbacks=callbacks)

#model.save('models/model_'+ str(args.modelnum) + '.h5')



