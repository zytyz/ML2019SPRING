XTEST=$1
DICT=$2
OUTPUT=$3
echo $XTEST
echo $DICT
echo $OUTPUT
mkdir data
python preprocess.py -xt $XTEST -dict $DICT
mkdir ans
cd ckpt
wget https://github.com/zytyz/tmp/releases/download/0.6.14/ckpt14.h5
wget https://github.com/zytyz/tmp/releases/download/0.6.14/ckpt15.h5
wget https://github.com/zytyz/tmp/releases/download/0.6.14/ckpt16.h5
wget https://github.com/zytyz/tmp/releases/download/0.6.14/ckpt17.h5
cd ..
python test.py -num 11 -test_data data/x_test_len50.npy
python test.py -num 12 -test_data data/x_test_len50.npy
python test.py -num 13 -test_data data/x_test_len30_dim50.npy
python test.py -num 14 -test_data data/x_test_embed_len50.npy
python test.py -num 15 -test_data data/x_test_embed_len50.npy
python test.py -num 16 -test_data data/x_test_embed_len30_dim50.npy
python test.py -num 17 -xt $XTEST -dict $DICT
python voting.py $OUTPUT