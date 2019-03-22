import numpy as np 
import pandas as pd
import argparse
import random as rd
import pathlib
import os

def process_command():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', default=0.001,type=float)
	parser.add_argument('-lamda', type=float,default=0)
	parser.add_argument('-epochs', default=50000,type=int)
	parser.add_argument('-num','--modelname',type=str)
	parser.add_argument('-use_val',action='store_true')
	parser.add_argument('-interval',default=1000,type=int)
	parser.add_argument('-reload',type=str)
	parser.add_argument('-batch_size',default=2048,type=int)
	parser.add_argument('-test',action='store_true')
	parser.add_argument('-norm_way',type=str,default='std')
	parser.add_argument('-thres','--threshold',type=float,default=0.5)
	parser.add_argument('-new_data',action='store_true')
	parser.add_argument('-model',type=str,default='log')
	parser.add_argument('-train_X_path',type=str)
	parser.add_argument('-train_Y_path',type=str)
	parser.add_argument('-train_raw_path',type=str)
	parser.add_argument('-test_X_path',type=str)
	parser.add_argument('-test_raw_path',type=str)
	parser.add_argument('-ans_path',type=str)
	args = parser.parse_args()
	return args

class ZData():
	def __init__(self,path_X,path_Y,args,norm_way):
		self.train_X = pd.read_csv(path_X)
		self.train_Y = pd.read_csv(path_Y)
		self.train_X = self.train_X.astype(np.float)
		self.train_Y = self.train_Y.astype(np.int)
		#if args.new_data:
			#self.train_X = self.modifydata(self.train_X)

		self.train_X = self.train_X.values
		self.train_Y = self.train_Y.values
		self.norm_way = norm_way
		if norm_way=='std':
			self.normalize()
		elif norm_way=='range':
			self.range_normalize()
		elif norm_way=='No':
			print('No normalization')
		self.add_bias()
		if args.use_val==True:
			self.split_data(args.modelname)

		print('train_X: {}'.format(self.train_X.shape))
		print('train_Y: {}'.format(self.train_Y.shape))
		if args.use_val==True:
			print('val_X: {}'.format(self.val_X.shape))
			print('val_Y: {}'.format(self.val_Y.shape))

	def getcount(self):
		ind_0 = []
		ind_1 = []
		for i in range(self.train_Y.shape[0]):
			if self.train_Y[i]==0:
				ind_0.append(i)
			elif self.train_Y[i]==1:
				ind_1.append(i)
		print('count0: {}, count1: {}'.format(len(ind_0),len(ind_1)))
		return ind_0,ind_1

	def modifydata(self,X):
		#X is a dataframe
		weird_col = X['fnlwgt']
		weird_col = weird_col.values
		'''print(weird_col)
		print('max: {}'.format(weird_col.max()))
		print('min: {}'.format(weird_col.min()))'''

		weird_col = (weird_col/10000).astype(np.int)

		print('max: {}'.format(weird_col.max()))
		print('min: {}'.format(weird_col.min()))

		print(' '.join([str(x) for x in np.ndarray.tolist(weird_col)[:10]]))

		tmp = []
		class_num = {str(x):0 for x in range(4)}
		for i in weird_col:
			x = np.zeros(4)
			classidx = int(i/10)
			if classidx>3:
				classidx=3
			class_num[str(classidx)]+=1
			x[classidx]=1
			tmp.append(x)
			
		print(class_num)
		weird_col = np.array(tmp).astype(np.int)

		#print(' '.join([str(x) for x in np.ndarray.tolist(weird_col)[:10]]))
		print(weird_col.shape)

		df = pd.DataFrame(weird_col)
		#print(df.head())

		total = pd.concat((X.drop(['fnlwgt'],axis=1),df),axis=1)
		#print(total)

		return total

	def normalize(self):
		self.mean = np.mean(self.train_X,axis=0)
		self.std = np.std(self.train_X,axis=0)
		self.train_X = (self.train_X-self.mean)/self.std

	def range_normalize(self):
		self.min = np.min(self.train_X,axis=0)
		self.max = np.max(self.train_X,axis=0)
		self.train_X = (self.train_X-self.min)/(self.max-self.min)

	def add_bias(self):
		self.train_X = np.concatenate((self.train_X,np.ones((self.train_X.shape[0],1))),axis=1)

	def split_data(self,modelname):
		val_len = int( 1/4 * self.train_Y.shape[0])
		ind = [ i for i in range(self.train_Y.shape[0])]
		#rd.shuffle(ind)
		print('ind for val: {}'.format(ind[:10]))
		ind_path = 'val_ind/val_ind_'+modelname+'.txt'
		with open(ind_path,'w') as f:
			f.write(' '.join([str(x) for x in ind]))
		self.val_X = self.train_X[ind[:val_len]]
		self.val_Y = self.train_Y[ind[:val_len]]
		self.train_X = self.train_X[ind[val_len:]]
		self.train_Y = self.train_Y[ind[val_len:]]

	def load_test(self,pathX):
		test_X = pd.read_csv(pathX).astype(np.float)
		#if args.new_data:
			#test_X = self.modifydata(test_X)
		test_X = test_X.values
		if self.norm_way=='std':
			test_X = (test_X-self.mean)/self.std
		elif self.norm_way=='range':
			test_X = (test_X-self.min)/(self.max-self.min)

		test_X = np.concatenate((test_X,np.ones((test_X.shape[0],1))),axis=1)
		print('test_X shape {}'.format(test_X.shape))
		return test_X


def reload_paras(path):
	with open(path) as f:
		paras = f.read().split(' ')
	paras = np.array(paras).astype(np.float).reshape(-1,1)
	return paras


def save_ans(Y,modelname):
	ans_path = args.ans_path
	if os.path.dirname(ans_path)!='': 
		if not os.path.isdir(os.path.dirname(ans_path)):
			dirname = os.path.dirname(ans_path)
			odir = pathlib.Path(dirname)
			odir.mkdir(parents=True, exist_ok=True)
	print('ans path: {}'.format(ans_path))
	labels=[]
	for i in Y:
		if i>0.5:
			labels.append(1)
		else:
			labels.append(0)
	df = pd.DataFrame({'id':[i for i in range(1,Y.shape[0]+1)], 'label':labels})
	df.to_csv(ans_path,index=False)


class Logistic_Regression():
	def __init__(self,paras=None):
		if paras is not None:
			print('model reloaded')
			self.W = paras
			self.best_W = paras
		else:
			self.W = np.zeros(train_X.shape[1]).reshape(-1,1)

	def sigmoid(self,Z):
		return 1 / (1 + np.exp(-Z))

	def predict(self,X):
		pre = self.sigmoid(np.matmul(X,self.W))
		#so that there won't be zeros or ones in predict
		return pre

	def label(self,Y):
		labels=[]
		for i in Y:
			if i>0.5:
				labels.append(1)
			else:
				labels.append(0)
		labels = np.array(labels).reshape(-1,1)
		return labels

	def eval(self,X,best=False):
		if best==True:
			pre = self.sigmoid(np.matmul(X,self.best_W))
		else:
			pre = self.sigmoid(np.matmul(X,self.W))

		return self.label(pre)
		

	def loss(self,targets,predict):
		l = np.sum(targets*np.log(predict+1e-9) + (1-targets)*(1-np.log(predict+1e-9)))
		return (-l)

	def accuracy(self,targets,labels):
		acc = sklearn.metrics.accuracy_score(targets,labels)
		return acc

	def save_model(self,path):
		paras = np.ndarray.tolist(self.W.reshape(-1))
		with open(path,'w') as f:
			f.write(' '.join([str(x) for x in paras]))

	def train_with_val(self,train_X,train_Y,val_X,val_Y,lr,epochs,modelname,interval,lamda):
		model_path = 'models/model_'+modelname+'.txt'
		best_model_path = 'best_models/best_model_'+modelname+'.txt'

		adagrad = np.zeros(train_X.shape[1]).reshape(-1,1)

		for epoch in range(epochs):
			pre_Y = self.predict(train_X)
			err =  pre_Y - train_Y
			gradient = np.matmul(np.transpose(train_X),err)+ 2*lamda*self.W
			adagrad += gradient**2
			#print(adagrad)
			self.W = self.W - lr*gradient/np.sqrt(adagrad)

			if epoch%interval==(interval-1):
				train_acc = self.accuracy(train_Y,self.label(pre_Y))
				val_acc = self.accuracy(val_Y,self.eval(val_X))
				#if epoch%100==99:
				print('epoch: {}, train accuracy: {}, val accuracy: {}'.format(epoch,train_acc,val_acc))
				self.save_model(model_path)
					
				try:
					if train_acc > best_train_acc:
						best_epoch = epoch
						best_train_acc = train_acc
						val_acc_in_best = val_acc
						self.best_W = self.W
						self.save_model(best_model_path)
				except:
					best_epoch = epoch
					best_train_acc = train_acc
					val_acc_in_best = val_acc
					self.best_W = self.W
					self.save_model(best_model_path)

		print('best epoch: {}'.format(best_epoch))
		print('best train acc: {} , val acc in best: {}'.format(best_train_acc,val_acc_in_best))
		'''if args.test==False:
			with open('handcraft_args.csv','a') as file:
				file.write(str(best_train_acc))
				file.write(',')
				file.write(str(best_epoch))
				file.write(',')
				file.write(str(val_acc_in_best))
				file.write(',')
				file.write(str(train_acc))
				file.write(',')
				file.write(str(val_acc))
				file.write('\n')'''


	def train(self,train_X,train_Y,lr,epochs,modelname,interval,lamda):
		model_path = 'models/model_'+modelname+'.txt'
		best_model_path = 'best_models/best_model_'+modelname+'.txt'

		self.W = np.zeros(train_X.shape[1]).reshape(-1,1)
		adagrad = np.zeros(train_X.shape[1]).reshape(-1,1)

		for epoch in range(epochs):
			pre_Y = self.predict(train_X)
			err =  pre_Y - train_Y
			gradient = np.matmul(np.transpose(train_X),err) + 2*lamda*self.W
			adagrad += gradient**2
			#print(adagrad)
			self.W = self.W - lr*gradient/np.sqrt(adagrad)

			if epoch%interval==(interval-1):
				train_acc = self.accuracy(train_Y,self.label(pre_Y))
				print('epoch: {}, train acc: {}'.format(epoch,train_acc))
				self.save_model(model_path)	
				try:
					if train_acc > best_train_acc:
						best_epoch = epoch
						best_train_acc = train_acc
						self.best_W = self.W
						self.save_model(best_model_path)
				except:
					best_epoch = epoch
					best_train_acc = train_acc
					self.best_W = self.W
					self.save_model(best_model_path)

		print('best epoch: {}'.format(best_epoch))
		print('best train acc: {}'.format(best_train_acc))
	
def save_args(args):
	#Name	lr	epochs	use_val	interval	
	#reload	batch_size	test	norm_way	
	#thres	new_data	model	use_sampler
	with open('handcraft_args.csv','a') as file:

		file.write(args.modelname)
		file.write(',')
		file.write(str(args.lr))
		file.write(',')
		file.write(str(args.epochs))
		file.write(',')
		file.write(str(args.use_val))
		file.write(',')
		file.write(str(args.lamda))
		file.write(',')
		file.write(str(args.interval))
		file.write(',')
		file.write(str(args.reload))
		file.write(',')
		file.write(str(args.batch_size))
		file.write(',')
		file.write(str(args.test))
		file.write(',')
		file.write(str(args.norm_way))
		file.write(',')
		file.write(str(args.threshold))
		file.write(',')
		file.write(str(args.new_data))
		file.write(',')
		file.write(str(args.model))
		file.write(',')

if __name__ == '__main__':
	args = process_command()
	print(args)
	#save_args(args)

	dt = ZData(args.train_X_path,args.train_Y_path,args,norm_way=args.norm_way)

	train_X, train_Y = dt.train_X,dt.train_Y
	#bias term is already added
	print('train_X raw shape: {}'.format(train_X.shape))
	print('train_Y raw shape: {}'.format(train_Y.shape))

	test_X = dt.load_test(args.test_X_path)
	print('test_X raw shape: {}'.format(test_X.shape))

	if args.reload==-1:
		model = Logistic_Regression()
	else:
		param = reload_paras('best_models/best_model_'+str(args.reload)+'.txt')
		model = Logistic_Regression(param)

	if args.test!=True:
		if args.use_val==True:
			val_X,val_Y = dt.val_X,dt.val_Y
			print('train_X shape: {}'.format(train_X.shape))
			print('train_Y shape: {}'.format(train_Y.shape))
			print('val_X shape: {}'.format(val_X.shape))
			print('val_Y shape: {}'.format(val_Y.shape))
			model.train_with_val(train_X,train_Y,val_X,val_Y,lr=args.lr,epochs=args.epochs,modelname=args.modelname,interval=args.interval,lamda=args.lamda)
		else:
			model.train(train_X,train_Y,lr=args.lr,epochs=args.epochs,modelname=args.modelname,interval=args.interval)

	test_Y = model.eval(test_X,best=True)

	save_ans(test_Y,args.modelname)




