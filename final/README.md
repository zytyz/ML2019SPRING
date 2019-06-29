
ML final
===

## requirements
1. python3.6
2. python requirements are in requirements.txt
3. matlab


## How to reproduce

dowload models and code:
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
