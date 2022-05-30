#!/bin/bash
#SBATCH --job-name=gpu-test    # Job name
#SBATCH --account=mdatascience_team
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dmccuske@umich.edu     # Where to send mail	
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=64gb
 
pwd; hostname; date
#echo "Running plot script on a single CPU core"
python wae_gan.py
date
