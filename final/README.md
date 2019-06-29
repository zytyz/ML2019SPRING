
ML final
===

## requirements
1. python3.6
2. python requirements are in requirements.txt
3. matlab

## Train
1. split Training images and Validation images into different directories
2. run the code (the result model will be in checkpoints/)
```bash
bash train.sh <Path to training hazy images> <Path to training GT images> <Path to testing hazy images> <Path to testing GT images> <Number of available GPUs>
```

## How to reproduce

download models and code:
```bash
bash download.sh
```

run the code (the result image will be in src/result):
```bash
bash reproduce.sh
```

compress into tgz for submission:
```bash
cd src/result
tar zcvf ans.tgz *.jpg
```
