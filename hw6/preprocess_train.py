#for submit
from preprocess import Preprocess
parser = argparse.ArgumentParser()
parser.add_argument('-xt',type=str,default='data/test_x.csv')
parser.add_argument('-xtr',type=str,default='data/train_x.csv')
parser.add_argument('-ytr',type=str,default='data/train_y.csv')
parser.add_argument('-dict',type=str,default='data/dict.txt.big')
args = parser.parse_args()

p = Preprocess(max_sentence_len=50,dim=128,train_x_path=args.xtr,train_y_path=args.ytr,test_path=args.xt,dict_path=args.dict)
p.getdata()
p = Preprocess(max_sentence_len=30,dim=50,train_x_path=args.xtr,train_y_path=args.ytr,test_path=args.xt,dict_path=args.dict)
p.getdata()