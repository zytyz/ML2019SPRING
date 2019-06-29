HAZY_TRAIN_PATH=$1
GT_TRAIN_PATH=$2
HAZY_VAL_PATH=$3
GT_VAL_PATH=$4
GPU_NUM=$5
python src/train/data_augment.py --size 224 --fold_A $HAZY_TRAIN_PATH --fold_B $GT_TRAIN_PATH --fold_AB TrainData
python src/train/data_augment.py --size 224 --fold_A $HAZY_VAL_PATH --fold_B $GT_VAL_PATH --fold_AB TestData 
mkdir logs
python src/train/train.py --train TrainData --test TestData --cuda --gpus $GPU_NUM --loss MSE --tag model 