
# Evaluation of MobileFaceNet on Glint360k

## Setup environment: 
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt 
```

## Run evaluation: 
```
cd ./recognition/arcface_torch
python3 validate.py configs/glint360k_mbf_2.py 
```