#!/bin/bash

python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/ImageCLEF/SymmNets/clef_train_c2i_cfg.yaml

python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/VisDA/SymmNets/visda17_train_train2val_cfg.yaml

