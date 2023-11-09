# dsa4266-gnome

## Prediction
After creating an ubuntu instance on AWS:
```
cd studies/ProjectStorage/
bash setup_linux.sh
python3 predict_main.py --test_sample # if you want to run prediction on the test dataset
python3 predict_main.py --sgnex # if you want to run prediction on SGNex data
```