#!/bin/bash
#SBATCH --job-name=rluam
#SBATCH --account=fc_rluam
#SBATCH --partition=savio2_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
## Command(s) to run:
module load python/3.11

# python -m venv ~/queueing

# Activate the virtual environment
source ~/queueing/bin/activate

# Install requirements
# pip install -r requirements.txt

python queue_runner_pd.py