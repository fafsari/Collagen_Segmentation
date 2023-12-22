#!/bin/sh
#SBATCH --qos=pinaki.sarder-b
#SBATCH --job-name=collagen_segmentation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100gb
#SBATCH --time=01:00:00
#SBATCH --output=collagen_seg_evaluation_%j.out

pwd; hostname; date
module load singularity

singularity exec ./collagen_segmentation_latest.sif python3 Collagen_Segmentation/CollagenEvaluate.py --test_model_path "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/Results/Ensemble_Green_Long" --label_path "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/C"

date