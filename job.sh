#!/bin/bash
#SBATCH --job-name=rluam
#SBATCH --account=fc_rluam
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
## Command(s) to run:
module load python/3.11

# python -m venv ~/queueing

# Activate the virtual environment
source ~/queueing/bin/activate

# Install requirements
pip install -r requirements.txt

python model.py