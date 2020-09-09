#!/usr/bin/env bash

data_path=$HOME/VisualSearch/muti_dataset/
run_id=0
config_name=efficientnet-b7-ns_lr5e-5decay0.8_finetune.json
train_collection=hunhe_withboss_val
val_collection=boss_val

GPU=$1
prefix=$2
python3 train_classifier.py \
--data_path $data_path \
--train_collection $train_collection \
--val_collection $val_collection \
--run_id $run_id \
--config_name $config_name \
--gpu $GPU \
--resume /data/dongchengbo/VisualSearch/muti_dataset/dfdc_dfv2_ff++_timit_bilibili_celeb5_opensoftware_gan_train/models/dfdc_dfv2_ff++_timit_bilibili_celeb5_opensoftware_gan_val/addmorereal/efficientnet-b7-ns_lr0.005decay0.8_multidata.json/run_0/model_5.pth.tar \
--seed 777 \
--prefix $prefix \
--workers 6 \
