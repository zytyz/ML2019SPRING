python submit_arrangedata.py $1
wget https://github.com/zytyz/tmp/releases/download/0.0.1/best_model_54.pkl
mkdir $2
python submit_saliency_map.py -num 54 -path $2
python submit_filter_vis.py -num 54 -path $2
python submit_lime.py -num 54 -path $2