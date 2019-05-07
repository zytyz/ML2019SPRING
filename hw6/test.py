import argparse
from keras.models import load_model
import numpy as np
from preprocess import Preprocess

parser = argparse.ArgumentParser()
parser.add_argument('-num','--modelnum',type=int)
parser.add_argument('-test_data',type=str,default='data/x_test_embed.npy')
args = parser.parse_args()
print(args)

model = load_model('ckpt/ckpt'+str(args.modelnum)+'.h5')
if args.modelnum==17:
	p = Preprocess(max_sentence_len=50,dim=128)
	x_test = p.gettestBOWdata()
elif args.modelnum==18:
	p = Preprocess(max_sentence_len=100,dim=128)
	_, x_test = p.getCHARdata()
else:
	x_test = np.load(args.test_data)

predict = model.predict(x_test)
y_test = (predict>0.5).astype(np.int)

with open('ans/ans_'+str(args.modelnum)+'.csv' , 'w') as f:
	f.write('id,label')
	f.write('\n')
	for i in range(y_test.shape[0]):
		f.write(str(i))
		f.write(',')
		f.write(str(y_test[i][0]))
		f.write('\n')