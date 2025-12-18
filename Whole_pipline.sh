#!/bin/bash
#SBATCH -J mkvlmdata
#SBATCH -o vlmdata.o%j
#SBATCH -t 120:00:00
#SBATCH -N 1 -n 8
#SBATCH --gres gpu:1
#SBATCH --mem=256GB

source activate agent

python Step1_agent.py
python Step2_agent.py
python Step3_calltools.py