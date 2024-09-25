#!/bin/sh
#SBATCH --qos=pinaki.sarder
#SBATCH --job-name=DAcollagen_segmentation
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40gb
#SBATCH --time=25:00:00
#SBATCH --output=logs/train_DA_V1.out

pwd; hostname; date
#module load singularity
module load conda
ml
date
#watch -n 1 nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhODE0Mzc0MC03MmQ5LTQ0NjYtOGU2OC1jZDQ1NGZhODUxNDAifQ==",
conda activate collagen_segment
python3 CollagenSegMain.py train_inputsPATH.json
#singularity exec --nv collagen_segmentation_latest.sif python3 CollagenSegMain.py train_inputs.json

date