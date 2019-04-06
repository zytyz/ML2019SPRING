import argparse
from keras.models import load_model
import numpy as np
import os
import pathlib
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-outpath',type=str)
parser.add_argument('-testpath',type=str)
parser.add_argument('-name','--modelname',type=str)
args = parser.parse_args()
print(args)

model = load_model(args.modelname)
test = pd.read_csv(args.testpath)

test_X = test['feature'].values
test_X = [ x.split(' ') for x in test_X]
test_X = np.array(test_X).astype(np.float)
test_X = test_X/255
#np.save('data/test_X.npy',test_X)
print(test_X.shape)

test_X = test_X.reshape(-1,48,48,1)

predict = model.predict(test_X)
test_Y = np.argmax(predict, axis = 1)

if os.path.dirname(args.outpath)!='': 
	if not os.path.isdir(os.path.dirname(args.outpath)):
		dirname = os.path.dirname(args.outpath)
		odir = pathlib.Path(dirname)
		odir.mkdir(parents=True, exist_ok=True)

with open(args.outpath , 'w') as f:
	f.write('id,label')
	f.write('\n')
	for i in range(test_Y.shape[0]):
		f.write(str(i))
		f.write(',')
		f.write(str(test_Y[i]))
		f.write('\n')