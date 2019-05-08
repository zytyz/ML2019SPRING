from gensim.models.word2vec import Word2Vec
import numpy as np 
import pandas as pd
import keras
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, Dropout, LSTM,Bidirectional
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import argparse
print(keras.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('-num','--modelnum',type=int)
parser.add_argument('-reload',type=int,default=-1)
args = parser.parse_args()

if args.modelnum==14 or args.modelnum==15:
	max_sentence_len = 50
	dim = 128
elif args.modelnum==16:
	max_sentence_len = 30
	dim = 50

if dim!=128:
	x_train = np.load('data/x_train_embed_len'+str(max_sentence_len)+'_dim'+str(dim)+'.npy')
	y_train = np.load('data/y_train.npy').reshape(-1,1)
	x_test = np.load('data/x_test_embed_len'+str(max_sentence_len)+'_dim'+str(dim)+'.npy')
else:
	x_train = np.load('data/x_train_embed_len'+str(max_sentence_len)+'.npy')
	y_train = np.load('data/y_train.npy').reshape(-1,1)
	x_test = np.load('data/x_test_embed_len'+str(max_sentence_len)+'.npy')


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

print('load embedded model...')

if dim!=128:
	word_model = Word2Vec.load('word_model/word2vec_dim'+str(dim)+'.bin')
else:
	word_model = Word2Vec.load('word_model/word2vec_new.bin')


pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape


if args.modelnum==16:
	#max_sentence_len = 30
	model = Sequential()
	model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
	model.add(Bidirectional(LSTM(units=dim, return_sequences=True)))
	model.add(Bidirectional(LSTM(units=dim,dropout=0.2,recurrent_dropout=0.2)))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(1, activation='sigmoid'))

if args.modelnum==15:
	#max_sentence_len = 50
	#dim = 128
	model = Sequential()
	model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
	model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
	model.add(Bidirectional(LSTM(units=128,dropout=0.2,recurrent_dropout=0.2)))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(1, activation='sigmoid'))


if args.modelnum==14:
	#max_sentence_len = 50
	#dim = 128
	model = Sequential()
	model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
	model.add(LSTM(128))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(64, activation='relu'))
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

