import numpy as np 
import pandas as pd
import keras
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, Dropout, LSTM,Bidirectional
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import argparse
from preprocess import Preprocess
print(keras.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('-num','--modelnum',type=int)
parser.add_argument('-reload',type=int,default=-1)
parser.add_argument('-xt',type=str,default='data/test_x.csv')
parser.add_argument('-xtr',type=str,default='data/train_x.csv')
parser.add_argument('-ytr',type=str,default='data/train_y.csv')
parser.add_argument('-dict',type=str,default='data/dict.txt.big')
args = parser.parse_args()

if args.modelnum==17:
	p = Preprocess(max_sentence_len=50,dim=128,train_x_path=args.xtr,train_y_path=args.ytr,test_path=args.xt,dict_path=args.dict)
	x_train, x_test = p.getBOWdata()
	y_train = np.load('data/y_train.npy').reshape(-1,1)

try:
    with open('val_ind/ind_list_'+str(args.reload)+'.txt') as f:
        ind_list = f.read().split(' ')
        ind_list = [int(x) for x in ind_list]
except:
    ind_list = [ i for i in range(y_train.shape[0])]
    import random as rd
    rd.shuffle(ind_list)
print('ind list {}'.format(ind_list[:10]))
val_len = int( 1/4 * y_train.shape[0])

path = 'val_ind/ind_list_'+str(args.modelnum)+'.txt'
with open(path,'w') as f:
    print('writing ind list in '+path)
    f.write(' '.join([str(x) for x in ind_list]))

x_val = x_train[ind_list[:val_len]]
y_val = y_train[ind_list[:val_len]]
x_train = x_train[ind_list[val_len:]]
y_train = y_train[ind_list[val_len:]]

print('train X shape {}'.format(x_train.shape))
print('train Y shape {}'.format(y_train.shape))
print('val X shape {}'.format(x_val.shape))
print('val Y shape {}'.format(y_val.shape))


if args.modelnum==17:
	#max_sentence_len = 30
	model = Sequential()
	model.add(Dense(1024, activation='relu', input_dim=x_train.shape[1]))
	model.add(Dropout(0.7))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

callbacks = []
callbacks.append(ModelCheckpoint('ckpt/ckpt'+str(args.modelnum)+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
csv_logger = CSVLogger('log/log'+str(args.modelnum)+'.csv', separator=',', append=False)
callbacks.append(csv_logger)
earlystop = EarlyStopping(monitor='val_acc', patience=10,mode='max')
callbacks.append(earlystop)
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), shuffle=True, batch_size=256, callbacks=callbacks)

