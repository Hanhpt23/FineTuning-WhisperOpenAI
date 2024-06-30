#!/bin/bash

#SBATCH --job-name=P       # create a short name for your job
#SBATCH --output=./gpu_output/prepare_dataset-JOB_ID_%j-%N.log # create output file

#SBATCH --nodes=1                  # node count
#SBATCH --ntasks-per-node=1        #CPUS per node (how many cpu to use withinin 1 node)
#SBATCH --mem=250G
#SBATCH --time=100:00:00               # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu2 --gres=gpu  # number of gpus per node


python3 -m src.data.prepare \
    --model_name "openai/whisper-tiny" \
    --data_name "Dataset/data" \
    --data_subset "default" \
    --language "de" \
    --do_lower_case False \
    --do_remove_punctuation False \
    --name_dataset_output "Hanhpt23/GermanMed" \
    --num_proc 1