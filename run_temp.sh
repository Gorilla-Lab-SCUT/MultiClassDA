#!/bin/bash
## The example script. all the log file are present in the ./experiments file for verifying

### ##################################### Closed Set DA example #########################################

## McDalNet script
#python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/ImageCLEF/McDalNet/clef_train_c2i_cfg.yaml
#
#python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/ImageCLEF/McDalNet/clef_train_c2i_cfg.yaml
#
#python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/ImageCLEF/McDalNet/clef_train_c2i_cfg.yaml
#
#python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/ImageCLEF/McDalNet/clef_train_c2i_cfg.yaml
#
#python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/ImageCLEF/McDalNet/clef_train_c2i_cfg.yaml

#python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/VisDA/McDalNet/visda17_train_train2val_cfg.yaml
#
#python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/VisDA/McDalNet/visda17_train_train2val_cfg.yaml
#
#python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/VisDA/McDalNet/visda17_train_train2val_cfg.yaml
#
#python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/VisDA/McDalNet/visda17_train_train2val_cfg.yaml
#
#python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/VisDA/McDalNet/visda17_train_train2val_cfg.yaml

## SymmNets script
#python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/ImageCLEF/SymmNets/clef_train_c2i_cfg.yaml

#python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/VisDA/SymmNets/visda17_train_train2val_cfg_res101.yaml --exp_name logres101

#python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/ImageCLEF/SymmNets/clef_train_c2i_cfg.yaml

#python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/VisDA/SymmNets/visda17_train_train2val_cfg.yaml


##################################### Partial DA example #######################
python ./tools/train.py  --distance_type None --method SymmNetsV2 --task partial --cfg ./experiments/configs/OfficeHome/SymmNets/home_train_A2R_partial_cfg.yaml

##################################### Open set DA example #####################
python ./tools/train.py  --distance_type None --method SymmNetsV2 --task open --cfg ./experiments/configs/Office31/SymmNets/office31_train_amazon2webcam_open_cfg.yaml