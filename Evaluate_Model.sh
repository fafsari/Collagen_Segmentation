#!/bin/sh
#SBATCH --job-name=glomerular_classification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --time=01:00:00
#SBATCH --output=model_evaluation_%j.out

pwd; hostname; date
module load singularity

## For Docker containers
image_name = "username/image:tag"
singularity exec docker//{$image_name} python3 evaluation.py --test_image_path "/path/to/test/images/" --output_path "/path/to/evaluation.csv"

date