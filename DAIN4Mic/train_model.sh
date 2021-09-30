#!/bin/bash -x

NUMEPOCH=1
BATCHSIZE=1
DATASETPATH='/content/DAIN/demo/spit_source_DAIN/128_img_separation/20210625_133002_prep_t_train'
LR=5e-06

#cd /content/DAIN
CUDA_VISIBLE_DEVICES=0 
python train.py \
       --datasetPath "${DATASETPATH}" \
       --numEpoch ${NUMEPOCH} \
       --batch_size ${BATCHSIZE} \
       --save_which 1 \
       --lr ${LR} \
       --rectify_lr 0.0005 \
       --flow_lr_coe 0.01 \
       --occ_lr_coe 0.0 \
       --filter_lr_coe 1.0 \
       --ctx_lr_coe 1.0 \
       --alpha 0.0 1.0 \
       --patience 4 \
       --factor 0.2



