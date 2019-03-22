import pandas as pd
import numpy as np
import argparse


def hash(column,classes=None):
	#print(classes)
	res = []
	for data in column:
		for i in range(len(classes)):
			if data in classes[i]:
				x = np.zeros(len(classes))
				x[i] = 1
				res.append(x)
				break
	res = np.array(res)
	return res,len(classes)

def hash_num(column,nums):
	interval = nums[0]
	classnum = nums[1]
	dic = {str((i+nums[2])*nums[0]):0 for i in range(classnum)}
	res = []
	for data in column:
		idx = int((data-nums[2])/interval)
		if idx >= classnum:
			idx = classnum -1
		if idx <0:
			idx=0
		x = np.zeros(classnum)
		x[idx]=1
		res.append(x)
		dic[str((idx+nums[2])*nums[0])]+=1
	print(dic)
		
	res = np.array(res)
	return res
	
def process_command():
	parser = argparse.ArgumentParser()
	parser.add_argument('-train_X_path',type=str)
	parser.add_argument('-train_Y_path',type=str)
	parser.add_argument('-train_raw_path',type=str)
	parser.add_argument('-test_X_path',type=str)
	parser.add_argument('-test_raw_path',type=str)
	parser.add_argument('-ans_path',type=str)
	parser.add_argument('-arr_train_X_path',type=str)
	parser.add_argument('-arr_test_X_path',type=str)
	args = parser.parse_args()
	return args

args = process_command()

df = pd.read_csv(args.train_raw_path)

#print(df.head())
#print(df.columns)

col_to_classes = {}

col_to_classes['age'] = None
col_to_classes['capital_gain'] = None
col_to_classes['capital_loss'] = None
col_to_classes['hours_per_week'] = None

#col_to_classes['marital_status'] = np.ndarray.tolist(np.unique(pd.Series.tolist(df['marital_status'])))
col_to_classes['marital_status'] = [[' Divorced',' Separated'],[' Married-AF-spouse',' Married-civ-spouse',' Married-spouse-absent'],[' Never-married'],[' Widowed']]
#col_to_classes['education'] = np.ndarray.tolist(np.unique(pd.Series.tolist(df['education'])))
col_to_classes['education'] = [[' 9th',' 10th', ' 11th', ' 12th'], [' 1st-4th' ],[' 5th-6th'],[ ' 7th-8th' ],
 [' Assoc-acdm' ,' Assoc-voc' ],[' Bachelors',' Some-college' ],[' Doctorate' ],[' HS-grad'],
 [' Masters'] ,[' Preschool'], [' Prof-school'] ]

col_to_classes['occupation'] = np.ndarray.tolist(np.unique(pd.Series.tolist(df['occupation'])))
col_to_classes['relationship'] = np.ndarray.tolist(np.unique(pd.Series.tolist(df['relationship'])))
col_to_classes['race'] = np.ndarray.tolist(np.unique(pd.Series.tolist(df['race'])))
col_to_classes['sex'] = np.ndarray.tolist(np.unique(pd.Series.tolist(df['sex'])))
#col_to_classes['native_country'] = np.ndarray.tolist(np.unique(pd.Series.tolist(df['native_country'])))
col_to_classes['native_country'] =[[' China',' Taiwan',' Hong',' Japan'],
	[' Vietnam',' Cambodia',' Laos',' India',' Thailand'],
	[' Canada'],
	[' Trinadad&Tobago',' Puerto-Rico',' Columbia',' Cuba',' Dominican-Republic',' Ecuador',' El-Salvador',' Guatemala',' Haiti',' Honduras',' Jamaica',' Nicaragua',' Peru'],
	[' Mexico',' Peru'],
	[' England',' France',' Germany',' Holand-Netherlands',' Italy',' Portugal',' Scotland'],
	[' Yugoslavia',' Greece',' Hungary',' Poland'],
	[' Outlying-US(Guam-USVI-etc)'],
	[' Iran'],
	[' Ireland'],
	[' Philippines'],
	[' ?'],
	[' South'],
	[' United-States']
	]

col_to_classes['workclass'] = np.ndarray.tolist(np.unique(pd.Series.tolist(df['workclass'])))
col_to_classes['fnlwgt'] = 100000,4,0
col_to_classes['education_num'] = 2,6,5

#print(col_to_classes)

for i in range(2):
	if i==1:
		df = pd.read_csv(args.test_raw_path)
	total = 0
	for key, value in col_to_classes.items():
		#print(key)
		col = pd.Series.tolist(df[key])
		if value is None:
			try:
				tmp = pd.DataFrame({key:col})
				total = pd.concat((total,tmp),axis=1)
			except:
				total = pd.DataFrame({key:col})
		elif type(value) is list:
			res,_ = hash(pd.Series.tolist(df[key]),value)
			tmp = pd.DataFrame({key+'_'+str(i):res[:,i] for i in range(len(value))})
			try:
				total = pd.concat((total,tmp),axis=1)
			except:
				total = tmp
		elif value == 'hash all':
			res,len_classes = hash(pd.Series.tolist(df[key]))
			tmp = pd.DataFrame({key+'_'+str(i):res[:,i] for i in range(len_classes)})
			total = pd.concat((total,tmp),axis=1)
		elif type(value) is tuple:
			res= hash_num(pd.Series.tolist(df[key]),value)
			tmp = pd.DataFrame({key+'_'+str(i):res[:,i] for i in range(value[1])})
			try:
				total = pd.concat((total,tmp),axis=1)
			except:
				total = tmp

	#print(total)

	if i==0:
		total.to_csv(args.arr_train_X_path,index=False)
	else:
		total.to_csv(args.arr_test_X_path,index=False)
