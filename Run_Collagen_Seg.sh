#!/bin/sh
#SBATCH --qos=pinaki.sarder
#SBATCH --job-name=collagen_segmentation
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50gb
#SBATCH --time=25:00:00
#SBATCH --output=logs/test_PATH.out

pwd; hostname; date
#module load singularity
module load conda
ml
date
#nvidia-smi

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhODE0Mzc0MC03MmQ5LTQ0NjYtOGU2OC1jZDQ1NGZhODUxNDAifQ==",

conda activate collagen_segment

#python3 CollagenSegMain.py train_inputs.json
python3 CollagenSegMain.py test_inputsPATH.json
#python3 CollagenSegMain.py test_inputsGeneral.json

date
