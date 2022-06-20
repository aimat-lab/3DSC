#!/bin/bash
#SBATCH --output=/pfs/work7/workspace/scratch/uoeci-3DSC/superconductors_3D/tmp_cluster_output/logs/%x_%j.out
#SBATCH --error=/pfs/work7/workspace/scratch/uoeci-3DSC/superconductors_3D/tmp_cluster_output/logs/%x_%j.err
#SBATCH --job-name=PyRun
#SBATCH --time=72:00:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=180000mb
#SBATCH --cpus-per-task=40

#SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
SCRIPT_DIR='/pfs/work7/workspace/scratch/uoeci-3DSC/superconductors_3D/superconductors_3D/utils/slurm'
cd $SCRIPT_DIR
echo "Changed working directory to $SCRIPT_DIR"

RUN="./run_in_env_superconductors_3D.sh"

echo "Running: "srun $RUN $@""

srun $RUN "$@"

