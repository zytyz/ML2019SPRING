import pandas as pd 
import numpy as np 
import sys

print('train path {}'.format(sys.argv[1]))
train = pd.read_csv(sys.argv[1])
print(train.head())

train_Y = train['label'].values.astype(np.int)
print(train_Y)
tmp = []
for i in train_Y:
	data = np.zeros(7)
	data[i] = 1
	tmp.append(data)
train_Y = np.array(tmp)
np.save('submit_train_Y.npy',train_Y)
print(train_Y.shape)
print(train_Y)

train_X = train['feature'].values
train_X = [ x.split(' ') for x in train_X]
train_X = np.array(train_X).astype(np.float)
train_X = train_X/255
np.save('submit_train_X.npy',train_X)
print(train_X.shape)
print(train_X)


'''test = pd.read_csv('data/test.csv')
print(test.head())

test_X = test['feature'].values
test_X = [ x.split(' ') for x in test_X]
test_X = np.array(test_X).astype(np.float)
test_X = test_X/255
np.save('data/test_X.npy',test_X)
print(test_X.shape)
print(test_X)
'''