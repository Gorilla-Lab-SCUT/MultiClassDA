#!/bin/bash

python ./tools/train.py  --distance_type None --method SymmNetsV2 --task closedsc --cfg ./experiments/configs/VisDA/SymmNets/visda17_train_train2val_cfg_res101SC.yaml --exp_name logCloseSC

python ./tools/train.py  --distance_type None --method SymmNetsV2 --task closedsc --cfg ./experiments/configs/VisDA/SymmNets/visda17_train_train2val_cfg_res101SC.yaml --exp_name logCloseSC






