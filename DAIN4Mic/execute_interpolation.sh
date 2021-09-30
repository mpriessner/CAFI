#!/bin/bash -x

TIMESTEP=0.5
MODEL=20166-Fri-May-08-07:43

#cd /content/DAIN
CUDA_VISIBLE_DEVICES=0 
python demo_MiddleBury_slowmotion.py \
       --predict "${MODEL}" \
       --netName DAIN_slowmotion \
       --time_step ${TIMESTEP}


