#!/bin/bash


ENV='superconductors_3D'

LOCALCONDA='/home/timo/anaconda3'
CLUSTERCONDA='/home/kit/stud/uoeci/conda'


if [ -d "$LOCALCONDA" ]; then
  CONDA="$LOCALCONDA"
else
  CONDA="$CLUSTERCONDA"
fi

__conda_setup="$("${CONDA}/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${CONDA}/etc/profile.d/conda.sh" ]; then
        . "${CONDA}/etc/profile.d/conda.sh"
    else
        export PATH="${CONDA}/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate $ENV

echo "Using conda environment ${ENV} from ${CONDA}."

