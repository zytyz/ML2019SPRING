import numpy as np 
import pandas as pd
import jieba
import emoji
from gensim.models.word2vec import Word2Vec
from multiprocessing import Pool
import argparse

class Preprocess():
	def __init__(self,max_sentence_len,dim,doc=None,train_x_path='data/train_x.csv',test_path='data/test_x.csv',train_y_path='data/train_y.csv',dict_path='data/dict.txt.big'):
		self.max_sentence_len = max_sentence_len
		self.dim = dim
		if doc is not None:
			self.doc = doc
		self.train_x_path = train_x_path
		self.test_path = test_path
		self.train_y_path = train_y_path
		jieba.load_userdict(dict_path) 

	def tokenize(self,sen):
		x = jieba.lcut(sen)
		x = [emoji.demojize(t) for t in x]
		return x

	def word2idx(self,word):
		return self.word_model.wv.vocab[word].index
	def idx2word(self,idx):
		return self.word_model.wv.index2word[idx]

	def sen2num(self,sen_ori):
		sen = []
		for tokenid in range(self.max_sentence_len):
			try:
				sen.append(self.word2idx(sen_ori[tokenid]))
			except:
				sen.append(self.word2idx('pad'))
		return sen

	def sen2vec(self,sen_ori):
		sen = []
		for tokenid in range(self.max_sentence_len):
			try:
				sen.append(self.word_model.wv[sen_ori[tokenid]])
			except:
				sen.append(self.word_model.wv['pad'])
		return sen

	def sen2BOW(self,sen_ori):
		bow = np.zeros(self.vocab_size)
		for tokenid in range(len(sen_ori)):
			try:
				bow[self.word2idx(sen_ori[tokenid])]+=1
			except:
				pass
		return bow

	def token2char(self,sen_ori):
		sen = [ch for ch in sen_ori]
		sen = [emoji.demojize(t) for t in sen]
		return sen

	def getCHARdata(self):
		x_train = np.array(pd.read_csv(self.train_x_path)['comment'])
		y_train = np.array(pd.read_csv(self.train_y_path)['label'])
		x_test = np.array(pd.read_csv(self.test_path)['comment'])

		print(x_train.shape)
		print(y_train.shape)
		print(x_test.shape)

		print('start splitting...')

		pool = Pool()
		x_train = pool.map(self.token2char,x_train)
		print(len(x_train))
		print(type(x_train))
		print('end tokenize train')

		x_test = pool.map(self.token2char,x_test)
		print(len(x_test))
		print(type(x_test))
		print('end tokenize test')

		try:
			self.word_model = Word2Vec.load('word_model/word2vec_char_dim'+str(self.dim)+'.bin')
		except:
			print('start training...')
			self.word_model = Word2Vec(x_train + x_test + [['unk','pad']],size=self.dim, window=5, min_count=1, workers=4, iter=100)
			print(len(x_train + x_test + [['unk','pad']]))
			print('saving word model...')
			self.word_model.save('word_model/word2vec_char_dim'+str(self.dim)+'.bin')
			print('train end')

		pretrained_weights = self.word_model.wv.syn0
		vocab_size, emdedding_size = pretrained_weights.shape
		self.vocab_size = vocab_size

		print('vocab size {}'.format(vocab_size))
		print('emdedding size {}'.format(emdedding_size))

		for word in ['女','子', '台','南', '學','校','白','痴','有','錢','unk','pad']:
		  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in self.word_model.most_similar(word)[:8])
		  print('  %s -> %s' % (word, most_similar))

		x_train_sen = x_train
		x_test_sen = x_test

		x_train = pool.map(self.sen2num,x_train_sen)
		x_train = np.array(x_train)
		print(x_train.shape)

		x_test = pool.map(self.sen2num,x_test_sen)
		x_test = np.array(x_test)	
		print(x_test.shape)

		return x_train, x_test

	def gettestBOWdata(self):
		x_test = np.array(pd.read_csv(self.test_path)['comment'])
		print(x_test.shape)

		print('start jieba...')

		pool = Pool()

		x_test = pool.map(self.tokenize,x_test)
		print(len(x_test))
		print(type(x_test))
		print('end tokenize test')
		mincount = 20

		try:
			self.word_model = Word2Vec.load('word_model/word2vec_dim'+str(self.dim)+'_mincount'+str(mincount)+'.bin')
			print('model loaded')
		except:
			print('cannot load model')
			print('start training...')
			self.word_model = Word2Vec(x_train + x_test + [['unk','pad']],size=self.dim, window=5, min_count=mincount, workers=4, iter=100)
			print(len(x_train + x_test + [['unk','pad']]))
			print('saving word model...')
			self.word_model.save('word_model/word2vec_dim'+str(self.dim)+'_mincount'+str(mincount)+'.bin')
			print('train end')

		pretrained_weights = self.word_model.wv.syn0
		vocab_size, emdedding_size = pretrained_weights.shape
		self.vocab_size = vocab_size

		print('vocab size {}'.format(vocab_size))
		print('emdedding size {}'.format(emdedding_size))

		x_test = pool.map(self.sen2BOW,x_test)
		x_test = np.array(x_test)
		print(x_test.shape)
		return x_test

	def getBOWdata(self):
		x_train = np.array(pd.read_csv(self.train_x_path)['comment'])
		y_train = np.array(pd.read_csv(self.train_y_path)['label'])
		x_test = np.array(pd.read_csv(self.test_path)['comment'])

		print(x_train.shape)
		print(y_train.shape)
		print(x_test.shape)

		print('start jieba...')

		pool = Pool()
		x_train = pool.map(self.tokenize,x_train)

		print(len(x_train))
		print(type(x_train))
		print('end tokenize train')

		x_test = pool.map(self.tokenize,x_test)
		print(len(x_test))
		print(type(x_test))
		print('end tokenize test')
		mincount = 20

		try:
			self.word_model = Word2Vec.load('word_model/word2vec_dim'+str(self.dim)+'_mincount'+str(mincount)+'.bin')
			print('model loaded')
		except:
			print('cannot load model')
			print('start training...')
			self.word_model = Word2Vec(x_train + x_test + [['unk','pad']],size=self.dim, window=5, min_count=mincount, workers=4, iter=100)
			print(len(x_train + x_test + [['unk','pad']]))
			print('saving word model...')
			self.word_model.save('word_model/word2vec_dim'+str(self.dim)+'_mincount'+str(mincount)+'.bin')
			print('train end')

		pretrained_weights = self.word_model.wv.syn0
		vocab_size, emdedding_size = pretrained_weights.shape
		self.vocab_size = vocab_size

		print('vocab size {}'.format(vocab_size))
		print('emdedding size {}'.format(emdedding_size))

		x_train_sen = x_train
		x_test_sen = x_test

		x_train = pool.map(self.sen2BOW,x_train_sen)
		x_train = np.array(x_train)
		print(x_train.shape)
		#np.save('data/x_train_BOW_len'+str(self.max_sentence_len)+'_dim'+str(self.dim)+'.npy',x_train)

		x_test = pool.map(self.sen2BOW,x_test_sen)
		x_test = np.array(x_test)
		print(x_test.shape)
		return x_train, x_test
		#np.save('data/x_test_BOW_len'+str(self.max_sentence_len)+'_dim'+str(self.dim)+'.npy',x_test)

	def gettestdata(self): #for submit
		x_test = np.array(pd.read_csv(self.test_path)['comment'])
		print(x_test.shape)

		print('start jieba...')
		pool = Pool()

		x_test = pool.map(self.tokenize,x_test)
		print(len(x_test))
		print(type(x_test))
		print('end tokenize test')

		try:
			if self.dim==128:
				self.word_model = Word2Vec.load('word_model/word2vec_new.bin')
			else:
				self.word_model = Word2Vec.load('word_model/word2vec_dim'+str(self.dim)+'.bin')
		except:
			print('start training...')
			self.word_model = Word2Vec(x_train + x_test + [['unk','pad']],size=self.dim, window=5, min_count=1, workers=4, iter=100)
			print(len(x_train + x_test + [['unk','pad']]))
			print('saving word model...')
			self.word_model.save('word_model/word2vec_dim'+str(self.dim)+'.bin')
			print('train end')

		pretrained_weights = self.word_model.wv.syn0
		vocab_size, emdedding_size = pretrained_weights.shape
		self.vocab_size = vocab_size

		print('vocab size {}'.format(vocab_size))
		print('emdedding size {}'.format(emdedding_size))

		x_test_sen = x_test

		x_test = pool.map(self.sen2num,x_test)
		x_test = np.array(x_test)
		print(x_test.shape)
		if self.dim!=128:
			np.save('data/x_test_embed_len'+str(self.max_sentence_len)+'_dim'+str(self.dim)+'.npy',x_test)
		else:
			np.save('data/x_test_embed_len'+str(self.max_sentence_len)+'.npy',x_test)

		x_test = pool.map(self.sen2vec,x_test_sen)
		x_test = np.array(x_test)	
		print(x_test.shape)
		#np.save('data/y_train.npy',y_train)
		if self.dim!=128:
			np.save('data/x_test_len'+str(self.max_sentence_len)+'_dim'+str(self.dim)+'.npy',x_test)
		else:
			np.save('data/x_test_len'+str(self.max_sentence_len)+'.npy',x_test)

		return x_test

	def getdata(self):
		x_train = np.array(pd.read_csv(self.train_x_path)['comment'])
		y_train = np.array(pd.read_csv(self.train_y_path)['label'])
		x_test = np.array(pd.read_csv(self.test_path)['comment'])

		print(x_train.shape)
		print(y_train.shape)
		print(x_test.shape)

		print('start jieba...')

		pool = Pool()
		x_train = pool.map(self.tokenize,x_train)

		print(len(x_train))
		print(type(x_train))
		print('end tokenize train')

		x_test = pool.map(self.tokenize,x_test)
		print(len(x_test))
		print(type(x_test))
		print('end tokenize test')

		try:
			if self.dim==128:
				self.word_model = Word2Vec.load('word_model/word2vec_new.bin')
			else:
				self.word_model = Word2Vec.load('word_model/word2vec_dim'+str(self.dim)+'.bin')
		except:
			print('start training...')
			self.word_model = Word2Vec(x_train + x_test + [['unk','pad']],size=self.dim, window=5, min_count=1, workers=4, iter=100)
			print(len(x_train + x_test + [['unk','pad']]))
			print('saving word model...')
			self.word_model.save('word_model/word2vec_dim'+str(self.dim)+'.bin')
			print('train end')

		pretrained_weights = self.word_model.wv.syn0
		vocab_size, emdedding_size = pretrained_weights.shape
		self.vocab_size = vocab_size

		print('vocab size {}'.format(vocab_size))
		print('emdedding size {}'.format(emdedding_size))

		for word in ['女子', '台南', '學校','白痴','有錢','unk','pad']:
		  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in self.word_model.most_similar(word)[:8])
		  print('  %s -> %s' % (word, most_similar))

		x_train_sen = x_train
		x_test_sen = x_test

		x_train = pool.map(self.sen2num,x_train)
		x_train = np.array(x_train)
		print(x_train.shape)
		if self.dim!=128:
			np.save('data/x_train_embed_len'+str(self.max_sentence_len)+'_dim'+str(self.dim)+'.npy',x_train)
		else:
			np.save('data/x_train_embed_len'+str(self.max_sentence_len)+'.npy',x_train)

		x_test = pool.map(self.sen2num,x_test)
		x_test = np.array(x_test)
		print(x_test.shape)
		if self.dim!=128:
			np.save('data/x_test_embed_len'+str(self.max_sentence_len)+'_dim'+str(self.dim)+'.npy',x_test)
		else:
			np.save('data/x_test_embed_len'+str(self.max_sentence_len)+'.npy',x_test)

		x_train = pool.map(self.sen2vec,x_train_sen)
		x_train = np.array(x_train)
		print(x_train.shape)
		if self.dim!=128:
			np.save('data/x_train_len'+str(self.max_sentence_len)+'_dim'+str(self.dim)+'.npy',x_train)
		else:
			np.save('data/x_train_len'+str(self.max_sentence_len)+'.npy',x_train)

		x_test = pool.map(self.sen2vec,x_test_sen)
		x_test = np.array(x_test)	
		print(x_test.shape)
		np.save('data/y_train.npy',y_train)
		if self.dim!=128:
			np.save('data/x_test_len'+str(self.max_sentence_len)+'_dim'+str(self.dim)+'.npy',x_test)
		else:
			np.save('data/x_test_len'+str(self.max_sentence_len)+'.npy',x_test)

		return x_train, y_train, x_test

	def convertDoc(self,method):
		x_test = np.array(self.doc)
		print('start jieba...')

		pool = Pool()
		x_test = pool.map(self.tokenize,x_test)
		print(len(x_test))
		print(type(x_test))
		print('end tokenize test')

		if method=='RNN':
			if self.dim==128:
				self.word_model = Word2Vec.load('word_model/word2vec_new.bin')
			else:
				self.word_model = Word2Vec.load('word_model/word2vec_dim'+str(self.dim)+'.bin')

			pretrained_weights = self.word_model.wv.syn0
			vocab_size, emdedding_size = pretrained_weights.shape
			self.vocab_size = vocab_size

			print('vocab size {}'.format(vocab_size))
			print('emdedding size {}'.format(emdedding_size))

			for word in ['女子', '台南', '學校','白痴','有錢','unk','pad']:
			  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in self.word_model.most_similar(word)[:8])
			  print('  %s -> %s' % (word, most_similar))

			x_test = pool.map(self.sen2num,x_test)
			x_test = np.array(x_test)
			print(x_test.shape)
		
			return x_test

		elif method=='BOW':
			mincount=20
			self.word_model = Word2Vec.load('word_model/word2vec_dim'+str(self.dim)+'_mincount'+str(mincount)+'.bin')
			
			pretrained_weights = self.word_model.wv.syn0
			vocab_size, emdedding_size = pretrained_weights.shape
			self.vocab_size = vocab_size

			print('vocab size {}'.format(vocab_size))
			print('emdedding size {}'.format(emdedding_size))

			for word in ['女子', '台南', '學校','白痴','有錢']:
			  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in self.word_model.most_similar(word)[:8])
			  print('  %s -> %s' % (word, most_similar))

			x_test_sen = x_test

			x_test = pool.map(self.sen2BOW,x_test)
			x_test = np.array(x_test)
			print(x_test.shape)
		
			return x_test

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-xt',type=str,default='data/test_x.csv')
	parser.add_argument('-xtr',type=str,default='data/train_x.csv')
	parser.add_argument('-ytr',type=str,default='data/train_y.csv')
	parser.add_argument('-dict',type=str,default='data/dict.txt.big')
	args = parser.parse_args()

	p = Preprocess(max_sentence_len=50,dim=128,train_x_path=args.xtr,train_y_path=args.ytr,test_path=args.xt,dict_path=args.dict)
	p.gettestdata()
	p = Preprocess(max_sentence_len=30,dim=50,train_x_path=args.xtr,train_y_path=args.ytr,test_path=args.xt,dict_path=args.dict)
	p.gettestdata()

