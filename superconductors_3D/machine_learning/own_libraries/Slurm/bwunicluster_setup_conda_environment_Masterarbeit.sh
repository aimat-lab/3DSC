#!/bin/bash

__conda_setup="$('/home/kit/stud/uoeci/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/kit/stud/uoeci/conda/etc/profile.d/conda.sh" ]; then
        . "/home/kit/stud/uoeci/conda/etc/profile.d/conda.sh"
    else
        export PATH="/home/kit/stud/uoeci/conda/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate Masterarbeit
