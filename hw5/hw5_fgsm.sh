OUTPATH=$2
INPATH=$1
echo $INPATH
echo $OUTPATH
mkdir $OUTPATH
python submit_fgsm.py -in_path $INPATH -out_path $OUTPATH