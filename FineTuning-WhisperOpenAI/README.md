Whisper Training

conda create -n Bi python=3.9
conda activate Bi

# Install dependent libraries
pip install -r requirement.txt

# Login to HuggingFace
huggingface-cli login ----> insert API

# Loggifn to wandb
wandb login ---> insert API


For training, run the file script
```slurm server
sbatch scripts/vn/base/vietmed-v1.sh
```