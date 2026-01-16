#!/bin/bash


cd /var/scratch/bacox/mobile-async-fl/
conda activate mobile_fl
module load cuda11.2/toolkit
python -c "import torch; print(torch.cuda.is_available())"
wandb login --verify