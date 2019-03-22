import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import os
import pathlib

def process_command():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', type=float)
	parser.add_argument('-epochs', type=int)
	parser.add_argument('-num','--modelname',type=str)
	parser.add_argument('-use_val',action='store_true')
	parser.add_argument('-interval',type=int)
	parser.add_argument('-reload',default=-1,type=int)
	parser.add_argument('-batch_size',default=2048,type=int)
	parser.add_argument('-test',action='store_true')
	parser.add_argument('-norm_way',type=str,default='std')
	parser.add_argument('-thres','--threshold',type=float,default=0.5)
	parser.add_argument('-new_data',action='store_true')
	parser.add_argument('-model',type=str,default='log')
	parser.add_argument('-use_sampler',action='store_true')
	parser.add_argument('-arr_train_X_path',type=str)
	parser.add_argument('-arr_test_X_path',type=str)
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
		#self.add_bias()
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
		print('test_X shape {}'.format(test_X.shape))
		return test_X


class ZDataset(Dataset):
	def __init__(self,train_X,train_Y):
		self.train_X = torch.from_numpy(train_X).float()
		self.train_Y = torch.from_numpy(train_Y).float()

	def __getitem__(self,idx):
		'''if self.train_Y[idx][0]==1:
			return self.train_X[idx], 0
		elif self.train_Y[idx][1]==1:
			return self.train_X[idx], 1'''
		return self.train_X[idx],self.train_Y[idx]

	def __len__(self):
		return self.train_Y.shape[0]

class Logistic_Model(nn.Module):
	def __init__(self,inputdim, outputdim):
		super(Logistic_Model, self).__init__() # call parent __init__ function
		self.linear = nn.Linear(inputdim,outputdim)
		self.sigmoid = nn.Sigmoid()
		self.inputdim = inputdim
	   
	def forward(self, x):
		# You can modify your model connection whatever you like
		out = self.linear(x.view(-1, self.inputdim))
		out = self.sigmoid(out)
		return out

class Deep_Model(nn.Module):
	def __init__(self,inputdim, outputdim):
		super(Deep_Model, self).__init__() # call parent __init__ function
		
		h1 = 1000
		h2 = 500
		h3 = 200

		self.fc = nn.Sequential(
			nn.Linear(inputdim, h1),
			nn.Sigmoid(),
			nn.Linear(h1, h2),
			nn.Sigmoid(),
			nn.Linear(h2, h3),
			nn.Sigmoid(),
			nn.Linear(h3, outputdim),
		)
		self.sigmoid = nn.Sigmoid()
		self.inputdim = inputdim
	   
	def forward(self, x):
		# You can modify your model connection whatever you like
		out = self.fc(x.view(-1, self.inputdim))
		out = self.sigmoid(out)
		return out

def getclass(target):
	label = []
	for i in target.view(-1):
		if i>args.threshold:
			label.append(1)
		else:
			label.append(0)
	label = torch.tensor(label).view(-1,1)
	return label


def train(model,optimizer,dataloader,loss_fn):
	model.train()
	train_loss = []
	train_acc = []
	for batch_num, (dataX,dataY) in enumerate(dataloader):

		dataX = Variable(dataX)
		dataY = Variable(dataY)

		optimizer.zero_grad()
		
		output = model(dataX)
		loss = loss_fn(output, dataY)
		loss.backward()
		optimizer.step()
		
		predict = getclass(output)
		acc = np.mean((dataY.long() == predict).numpy())
		
		train_acc.append(acc)
		train_loss.append(loss.item())
	return np.mean(train_loss), np.mean(train_acc)

def test(model,valloader):
	model.eval()
	val_acc = []
	with torch.no_grad():
		for _, (dataX,dataY) in enumerate(valloader): # Under `torch.no_grad()`, no need to wrap data & target in `Variable`
			output = model(dataX)
			predict = getclass(output)
			acc = np.mean((dataY.long() == predict).numpy())
			val_acc.append(acc)
	return np.mean(val_acc)
	#print("Epoch: {}, Acc: {:.4f}".format(epoch, np.mean(val_acc)))

def save_ans(Y,path):
	ans_path = path
	if os.path.dirname(path)!='': 
			if not os.path.isdir(os.path.dirname(path)):
				dirname = os.path.dirname(path)
				odir = pathlib.Path(dirname)
				odir.mkdir(parents=True, exist_ok=True)
	print('ans path: {}'.format(ans_path))
	labels = Y.squeeze().tolist()
	df = pd.DataFrame({'id':[i for i in range(1,Y.shape[0]+1)], 'label':labels})
	df.to_csv(ans_path,index=False)

def save_args(args):
	#Name	lr	epochs	use_val	interval	
	#reload	batch_size	test	norm_way	
	#thres	new_data	model	use_sampler
	with open('args.csv','a') as file:

		file.write(args.modelname)
		file.write(',')
		file.write(str(args.lr))
		file.write(',')
		file.write(str(args.epochs))
		file.write(',')
		file.write(str(args.use_val))
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
		file.write(str(args.use_sampler))
		file.write(',')

if __name__=='__main__':
	args = process_command()
	print(args)
	#save_args(args)

	if args.new_data==True:
		trainpath = args.arr_train_X_path
		testpath = args.arr_test_X_path
	else:
		trainpath = args.train_X_path
		testpath = args.test_X_path

	dt = ZData(trainpath,args.train_Y_path,args,norm_way=args.norm_way)

	dataset = ZDataset(dt.train_X,dt.train_Y)

	if args.test==False:
		if args.use_sampler==True:
			class_sample_count = dt.getcount()
			weights = np.zeros(dt.train_Y.shape[0])
			weights[class_sample_count[0]]=0.3
			weights[class_sample_count[1]]=0.7
			print(weights)

			sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(dataset),replacement=True)
			dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
		else:
			dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

		if args.use_val==True:
			valdataset = ZDataset(dt.val_X,dt.val_Y)
			valloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

	if args.model=='log':
		model = Logistic_Model(inputdim=dt.train_X.shape[1],outputdim=1)
	elif args.model=='deep':
		model = Deep_Model(inputdim=dt.train_X.shape[1],outputdim=1)

	if args.reload!=-1:
		model = torch.load('best_models/best_model_'+str(args.reload)+'.pkl')

	if args.test==False:
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		loss_fn = nn.BCELoss()

		best_val_acc=0
		for epoch in range(args.epochs):
			train_loss,train_acc = train(model,optimizer,dataloader,loss_fn)
			torch.save(model,'models/model_'+args.modelname+'.pkl')
			if args.use_val==True:
				val_acc = test(model,valloader)
			if val_acc>best_val_acc:
				best_epoch = epoch
				best_val_acc = val_acc
				train_acc_in_best = train_acc
				torch.save(model,'best_models/best_model_'+args.modelname+'.pkl')
			if args.use_val==True:
				print('Epoch: {}, train loss {:.10f}, train acc {:10f}, val acc {:10f}'.format(epoch,train_loss,train_acc,val_acc))
			else:
				print('Epoch: {}, train loss {:.10f}, train acc {:10f}'.format(epoch,train_loss,train_acc))
		if args.use_val==True:
			print('best epoch:{}, best val acc:{}, train acc in best: {}'.format(best_epoch,best_val_acc,train_acc_in_best))

	'''if args.test==False:
		with open('args.csv','a') as file:
			file.write(str(best_val_acc))
			file.write(',')
			file.write(str(best_epoch))
			file.write(',')
			file.write(str(train_acc_in_best))
			file.write(',')
			file.write(str(train_acc))
			file.write(',')
			file.write(str(val_acc))
			file.write('\n')'''

	model.eval()
	model = torch.load('best_models/best_model_'+str(args.reload)+'.pkl')
	test_X = dt.load_test(testpath)
	test_X = torch.from_numpy(test_X).float()
	print(test_X.shape)
	test_Y = model(test_X)
	test_Y = getclass(test_Y)
	
	
	save_ans(test_Y,args.ans_path)

