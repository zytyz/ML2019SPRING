import pandas as pd 
import numpy as np
import sys
import os
import pathlib

anss = [11,12,13,14,15,16,17]
total = pd.DataFrame({'id':range(20000)})
for i in anss:
	total['y'+str(i)] = pd.read_csv('ans/ans_'+str(i)+'.csv')['label'].values

total['ans'] = [0]*len(total)
for i in anss:
	total['ans']+=total['y'+str(i)]

on_thres = (total['ans'].values == len(anss)/2).astype(np.int)
for idx in range(len(on_thres)):
	if on_thres[idx]==1:
		print(idx)

print(on_thres.sum())

y_test = (total['ans'].values >= (len(anss)/2) ).astype(np.int)

if os.path.dirname(sys.argv[1])!='': 
	if not os.path.isdir(os.path.dirname(sys.argv[1])):
		dirname = os.path.dirname(sys.argv[1])
		odir = pathlib.Path(dirname)
		odir.mkdir(parents=True, exist_ok=True)

with open(sys.argv[1], 'w') as f:
	f.write('id,label')
	f.write('\n')
	for i in range(y_test.shape[0]):
		f.write(str(i))
		f.write(',')
		f.write(str(y_test[i]))
		f.write('\n')