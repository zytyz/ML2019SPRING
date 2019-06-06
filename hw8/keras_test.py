import argparse
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, DepthwiseConv2D,BatchNormalization, LeakyReLU, AveragePooling2D,GlobalAvgPool2D,Dropout
import pandas as pd 
import argparse
from multiprocessing import Pool
from keras.models import load_model

def getimgs(raw):
    img = np.array(raw.split(' ')).reshape(48,48,1)
    return img

def readfile(path):
    print("Reading File...")
    raw_test = pd.read_csv(path)['feature'].values
    print('raw test shape {}'.format(raw_test.shape))

    p = Pool()
    x_test = p.map(getimgs,raw_test)
    x_test = np.array(x_test,dtype=float)
    x_test /= 255.0

    print(x_test.shape)
    print(x_test.shape[0])

    print('x_test {}'.format(x_test.shape))
    return x_test

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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-testpath',default='data/test.csv',type=str)
parser.add_argument('-num','--modelnum',default=0,type=int,required=True)
parser.add_argument('-drop',action='store_true')
parser.add_argument('-ori_model',action='store_true')
parser.add_argument('-com',action='store_true')
parser.add_argument('-outpath',type=str)
args = parser.parse_args()
print(args)

x_test = readfile(args.testpath)


if args.com:
	loaded = np.load('wcom_'+str(args.modelnum)+'.npz')
	w = []
	for arr in loaded.files:
	    w.append(loaded[arr])
else:
	w = np.load('weights_'+str(args.modelnum)+'.npy',allow_pickle=True)

if args.modelnum==101:
	assert args.drop==True
	img = Input(shape=(48,48,1))
	x = conv_block(img,32,(2,2))
	x = dw_block(x,64,(1,1))
	x = dw_block(x,128,(2,2))
	x = dw_block(x,128,(1,1))
	x = dw_block(x,128,(1,1))
	x = dw_block(x,128,(1,1))
	x = dw_block(x,128,(1,1))
	x = GlobalAvgPool2D()(x)
	if args.drop:
	    x = Dropout(0.5)(x)
	x = Dense(7,activation='softmax')(x)
	model = Model(inputs=img,outputs=x)

elif args.modelnum==104:
	assert args.drop==True
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
model.set_weights(w)

#if args.ori_model:
	#model = load_model('ckpt/ckpt'+str(args.modelnum)+'.h5')

predict = model.predict(x_test)
y_test = np.argmax(predict, axis = 1)

if args.outpath not None:
	path = args.outpath
else:
	path = 'ans/ans_'+str(args.modelnum)+'.csv' 

with open( path , 'w') as f:
	f.write('id,label')
	f.write('\n')
	for i in range(y_test.shape[0]):
		f.write(str(i))
		f.write(',')
		f.write(str(y_test[i]))
		f.write('\n')

