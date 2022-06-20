#!/bin/bash
#SBATCH --output=/home/kit/stud/uoeci/superconductors_3D/tmp_cluster_output/logs/%x_%j.out
#SBATCH --error=/home/kit/stud/uoeci/superconductors_3D/tmp_cluster_output/logs/%x_%j.err
#SBATCH --job-name=PyRun
#SBATCH --time=72:00:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=180000mb
#SBATCH --cpus-per-task=40

RUN='/home/kit/stud/uoeci/superconductors_3D/superconductors_3D/utils/slurm/run_in_env.sh'

echo "Running "$@""

srun $RUN "$@"

