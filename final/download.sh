#!/bin/bash
#wget --no-chick-certificate -r 'https://docs.google.com/uc?export=download&id=1x-B82Lk4bim2HQghDf7adVg8uobOsAnt' -O src.zip
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id=1x-B82Lk4bim2HQghDf7adVg8uobOsAnt' -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O src.zip \
     'https://docs.google.com/uc?export=download&id=1x-B82Lk4bim2HQghDf7adVg8uobOsAnt&confirm='$(<confirm.txt)
rm -f confirm.txt
rm -f cookies.txt
unzip src.zip
rm -f src.zip
