#!/bin/bash
## Set max runtime (24h)
#SBATCH --time=120:00:00
## Set parallelism (3)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

## Set NVidia A4000 (16GB) GPU 
## Make CUDA device visible to host
echo "First output"

## Host envirionmental variables
export CUDA_VERSION="11.2"

## Singularity setup variables
## Default working directory of dis/pytorch-base:2.2.1+
export WORK_DIR=/workspace
export IMAGE_PATH=/var/scratch/$USER


# Mount points and variables
## Note that we export the variables, s.t. they are accessible on the node once `srun` is invoked





# Load required modules
## Add option to load
source /opt/ohpc/admin/lmod/lmod/init/profile
## Run
module load "cuda${CUDA_VERSION}/toolkit/${CUDA_VERSION}"
# module load "cuda${CUDA_VERSION}/toolkit"

# Determine working order
# source ./obtain_lead.sh
# source /home/$USER/obtain_lead.sh
cd /var/scratch/bacox/mobile-async-fl/
conda init
conda activate mobile_fl

mkdir -p out/python

echo "World size = $SLURM_NTASKS"

# for i in {1..$#node_array}; do
#        printf  "%s\n" "$node_array[i] slots=1"  >> ./config/nodelist_${SLURM_JOB_ID}
# done

echo "Running exp $1"

# zsh -c "blablabla ${SLURM_PROCID}"
srun --output="out/test_run_${SLURM_JOB_ID}.%N.%t.out" \
       --error="out/test_run_${SLURM_JOB_ID}.%N.%t.err" \
       --jobid $SLURM_JOBID \
       bash -c "bash scripts/seq_task.sh $1" 
       # bash -c "echo ${SLURM_PROCID} && python3 hello_world.py $SLURM_PROCID" 
#  python -u hello_world.py $SLURM_PROCID
