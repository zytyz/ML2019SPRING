TEST_PATH=$1
OUTPUT_PATH=$2
echo test path $TEST_PATH
echo output path $OUTPUT_PATH
wget https://github.com/zytyz/tmp/releases/download/0.0.0/ckpt_4.h5
python submit_test.py -name ckpt_4.h5 -outpath $OUTPUT_PATH -testpath $TEST_PATH