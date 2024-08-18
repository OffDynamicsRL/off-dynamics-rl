#!/bin/bash

# we give some examples on running the experiments

for ((i=1;i<6;i+=1))
do 
    # online-online example: DARC
    CUDA_VISIBLE_DEVICES=0 python train.py --policy DARC --env halfcheetah-kinematic-footjnt --shift_level easy --seed $i --dir logs &
    CUDA_VISIBLE_DEVICES=0 python train.py --policy DARC --env hopper-gravity --shift_level 0.5 --seed $i --dir logs &

    ## offline-online example: BC_SAC
    CUDA_VISIBLE_DEVICES=1 python train.py --policy BC_SAC --env halfcheetah-friction --shift_level 0.1 --seed $i --mode 1 --srctype medium --dir logs &
    CUDA_VISIBLE_DEVICES=1 python train.py --policy BC_SAC --env pen-broken-joint --shift_level easy --seed $i --mode 1 --srctype expert --dir logs &
    
    # online-offline example
    CUDA_VISIBLE_DEVICES=2 python train.py --policy SAC_MCQ --env walker2d-gravity --shift_level 2.0 --seed $i --mode 2 --tartype medium --dir logs &
    CUDA_VISIBLE_DEVICES=2 python train.py --policy SAC_MCQ --env ant-morph-alllegs --shift_level medium --seed $i --mode 2 --tartype medium --dir logs &
    
    # offline-offline example
    CUDA_VISIBLE_DEVICES=3 python train.py --policy BOSA --env ant-friction --shift_level 0.5 --seed $i --mode 3 --srctype medium --tartype medium --dir logs &
    CUDA_VISIBLE_DEVICES=3 python train.py --policy BOSA --env hopper-morph-torso --shift_level medium --seed $i --mode 3 --srctype medium --tartype medium --dir logs &
done
