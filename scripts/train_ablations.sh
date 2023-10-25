#!/bin/bash

NUM_GPUS=3
JOB_NUM=0

for RANDOM_SEED in 0 1 2 3 4 5 6 7 8 9
do
  EXP_FOLDER="clips/exp_oct25_seed_$RANDOM_SEED"
  echo $EXP_FOLDER

  ## MATCH
  CUDA_VISIBLE_DEVICES=$(($JOB_NUM % $NUM_GPUS)) python train.py train.model=single_cls train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=50 train.aggregator.type=two_random_index train.load_from_last_ckpt=False &
  JOB_NUM=$((JOB_NUM + 1))
  CUDA_VISIBLE_DEVICES=$(($JOB_NUM % $NUM_GPUS)) python train.py train.model=rotator train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=50 train.aggregator.type=two_random_index train.load_from_last_ckpt=False &
  JOB_NUM=$((JOB_NUM + 1))
  CUDA_VISIBLE_DEVICES=$(($JOB_NUM % $NUM_GPUS)) python train.py train.model=true_rotator train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=50 train.aggregator.type=two_random_index train.load_from_last_ckpt=False &
  JOB_NUM=$((JOB_NUM + 1))
done
