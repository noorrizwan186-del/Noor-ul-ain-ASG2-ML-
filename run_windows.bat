@echo off
python -m pip install -r requirements.txt
python train_models.py
python -m uvicorn main:app --reload
