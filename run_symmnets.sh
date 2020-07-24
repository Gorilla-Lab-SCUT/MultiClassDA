#!/bin/bash

################ script for the DomainNet dataset.

#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_c2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_c2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_c2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_c2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_c2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_c2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_c2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_c2p_cfg.yaml
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2c_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2i_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2q_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2r_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_p2s_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2c_cfg.yaml

#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2i_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2p_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2q_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV1 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2s_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_r2s_cfg.yaml
#
#
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2r_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2r_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2c_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2c_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2i_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2i_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2p_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2p_cfg.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type L1 --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type KL --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type CE --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type MDD --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type DANN --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train.py  --distance_type None --method SymmNetsV2 --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2q_cfg.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3  python ./tools/train.py  --distance_type SourceOnly --method McDalNet --cfg ./experiments/configs/DomainNet/McDalNet/domain_train_s2q_cfg.yaml