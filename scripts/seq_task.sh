#!/bin/bash
conda init
conda activate mobile_fl

echo "Running python3 -m mobilefl $1 --print --wandb"
# python3 delayed.py $1 $SLURM_JOBID
python3 -m mobilefl $1 --print --wandb