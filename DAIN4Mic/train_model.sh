#!/bin/bash -x

NUMEPOCH=1
BATCHSIZE=1
DATASETPATH='/content/gdrive/My Drive/1.2_BIG_DATA_PhD_Project_2/3.DAIN/_SIMULATED_DATA_TEST/Master_test/t_split/Set_1'

#cd /content/DAIN
CUDA_VISIBLE_DEVICES=0 
python train.py \
       --datasetPath "${DATASETPATH}" \
       --numEpoch ${NUMEPOCH} \
       --batch_size ${BATCHSIZE} \
       --save_which 1 \
       --lr 0.0005 \
       --rectify_lr 0.0005 \
       --flow_lr_coe 0.01 \
       --occ_lr_coe 0.0 \
       --filter_lr_coe 1.0 \
       --ctx_lr_coe 1.0 \
       --alpha 0.0 1.0 \
       --patience 4 \
       --factor 0.2



