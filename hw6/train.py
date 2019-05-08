import numpy as np 
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout,Bidirectional
from keras.callbacks import CSVLogger, ModelCheckpoint,EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-num','--modelnum',type=int)
parser.add_argument('-reload',type=int,default=-1)
args = parser.parse_args()

if args.modelnum==11 or args.modelnum==12:
	max_sentence_len = 50
	dim = 128
elif args.modelnum==13:
	max_sentence_len = 30
	dim = 50

if dim!=128:
	x_train = np.load('data/x_train_len'+str(max_sentence_len)+'_dim'+str(dim)+'.npy')
	y_train = np.load('data/y_train.npy').reshape(-1,1)
	x_test = np.load('data/x_test_len'+str(max_sentence_len)+'_dim'+str(dim)+'.npy')
else:
	x_train = np.load('data/x_train_len'+str(max_sentence_len)+'.npy')
	y_train = np.load('data/y_train.npy').reshape(-1,1)
	x_test = np.load('data/x_test_len'+str(max_sentence_len)+'.npy')

print('x_train shape {}'.format(x_train.shape))
print('y_train shape {}'.format(y_train.shape))
print('x_test shape {}'.format(x_test.shape))
print('\n')



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

model = Sequential()

if args.modelnum==13:
	#max_sentence_len = 30
	#dim = 50
	model.add(Bidirectional(LSTM(units=dim, return_sequences=True),input_shape=(max_sentence_len, dim)))
	model.add(Bidirectional(LSTM(units=dim,dropout=0.2,recurrent_dropout=0.2)))
	model.add(Dense(units=64,activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(units=32,activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(units=1,activation='sigmoid'))

if args.modelnum==12:
	#max_sentence_len = 50
	model.add(Bidirectional(LSTM(units=128, return_sequences=True),input_shape=(max_sentence_len, 128)))
	model.add(Bidirectional(LSTM(units=128,dropout=0.2,recurrent_dropout=0.2)))
	model.add(Dense(units=128,activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(units=64,activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(units=1,activation='sigmoid'))


if args.modelnum==11:
	#max_sentence_len = 50
	model.add(LSTM(units=128,input_shape=(max_sentence_len,128)))
	model.add(Dense(units=128,activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(units=64,activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(units=1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


callbacks = []
callbacks.append(ModelCheckpoint('ckpt/ckpt'+str(args.modelnum)+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
csv_logger = CSVLogger('log/log'+str(args.modelnum)+'.csv', separator=',', append=False)
callbacks.append(csv_logger)
earlystop = EarlyStopping(monitor='val_acc', patience=10,mode='max')
callbacks.append(earlystop)


model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), shuffle=True, batch_size=256, callbacks=callbacks)




