#!/bin/sh
#SBATCH --qos=pinaki.sarder-b
#SBATCH --job-name=collagen_segmentation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100gb
#SBATCH --time=01:00:00
#SBATCH --output=collagen_seg_evaluation_%j.out

pwd; hostname; date
module load singularity

singularity exec ./collagen_segmentation_latest.sif python3 Collagen_Segmentation/CollagenEvaluate.py --test_model_path "/blue/pinaki.sarder/samuelborder/Same_Training_Set/Brightfield_G" --label_path "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/C" --train_test_names "/blue/pinaki.sarder/samuelborder/Same_Training_Set/Brightfield_G/Merged_Results_Table.csv"

date