#!/bin/bash

ENV="./setup_env.sh"

# Load conda environment
source $ENV
echo 'Using python:'
which python
sleep 1

# Load software modules
module load devel/cuda/11.0
echo 'Loaded module devel/cuda/11.0'

python "$@"

