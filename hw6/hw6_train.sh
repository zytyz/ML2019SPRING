XTRAIN=$1
YTRAIN=$2
XTEST=$3
DICT=$4
mkdir data
python preprocess_train.py -xt $XTEST -xtr $XTRAIN -ytr $YTRAIN -dict $DICT
mkdir ckpt
mkdir log
python train.py -num 11 -reload 11
python train.py -num 12 -reload 12
python train.py -num 13 -reload 13
python train_with_embed.py -num 14 -reload 14
python train_with_embed.py -num 15 -reload 15
python train_with_embed.py -num 16 -reload 16
python train_BOW.py -num 17 -reload 17 -xt $XTEST -ytr $YTRAIN -xtr $XTRAIN -dict $DICT