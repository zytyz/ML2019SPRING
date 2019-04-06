TRAIN_PATH=$1
python submit_arrangedata.py $TRAIN_PATH
mkdir submit_checkpoints
mkdir submit_csv_log
python submit_train.py