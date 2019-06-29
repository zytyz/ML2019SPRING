#!/bin/bash

cd ./src/indoor
bash run_dehaze.sh
cd ./indoor
matlab -nodisplay -nodesktop -nosplash -r "run demo.m; exit;"
cd ../../outdoor
bash run_dehaze.sh
cd ./outdoor
matlab -nodisplay -nodesktop -nosplash -r "run demo.m; exit;"
cd ../..
mkdir tmp
cp ./indoor/indoor/our_cvprw_submitted/*.png ./tmp
cp ./outdoor/outdoor/our_cvprwoutdoor_submitted/*.png ./tmp
python3.6 gen_result.py
rm -R -f tmp
python3.6 hist.py

