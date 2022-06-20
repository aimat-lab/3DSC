#!/bin/bash

# Parse command line options.
source "${HOME}/Masterarbeit/Rechnungen/Skripte/helperscripts/bwunicluster_setup_conda_environment_Masterarbeit.sh"
echo 'Using python:'
which python
sleep 1

python "$@"

