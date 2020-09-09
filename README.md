# Get Landmarks

## Environment

+ Ubuntu 16.04.6 LTS
+ Python 3.5.2

## Requirement
+ Install nvidia-apex follow https://github.com/NVIDIA/apex, move it here.
+ pip install requirements.txt

## Usage


### Training
Change following in run_train.sh
```
data_path, run_id, config_name, train_collection, val_collection, test_collection
```
Then run 
```bash 
bash run_train.sh gpu_id additional_description
e.g. bash run_train.sh 0 Vfinetune
```
### Prediction
for img: 

```bash
bash run_pred_img.sh gpu_id if_submision
e.g. bash run_pred_img.sh 0 1 
```
