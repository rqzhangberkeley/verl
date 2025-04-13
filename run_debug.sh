#!/bin/bash
source /home/jovyan/project/miniconda/etc/profile.d/conda.sh
conda activate /home/jovyan/project/miniconda/envs/verl
python /home/jovyan/project/verl/verl/trainer/main_ppo.py "$@" 