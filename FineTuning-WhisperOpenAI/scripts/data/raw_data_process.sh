#!/bin/bash

#SBATCH --job-name=P       # create a short name for your job
#SBATCH --output=./gpu_output/Raw_prepare_dataset-JOB_ID_%j-%N.log # create output file

#SBATCH --nodes=1                  # node count
#SBATCH --ntasks-per-node=1        #CPUS per node (how many cpu to use withinin 1 node)
#SBATCH --mem=250G
#SBATCH --time=100:00:00               # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu1 --gres=gpu  # number of gpus per node

echo "Job ID: $SLURM_JOBID"
echo "Node names: $SLURM_JOB_NODELIST"

python3 -m src.data.hanh_process \
    --train_scription_path my_data/train.json \
    --test_scription_path my_data/test.json \
    --val_scription_path my_data/val.json \
    --name_push "Hanhpt23/GermanMed" 

echo "Job ID: $SLURM_JOBID Done!!!!"
