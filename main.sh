#!/bin/bash
#SBATCH -J Process_Nimbus
#SBATCH -o Process_Nimbus.o%j
#SBATCH -t 20:00:00
#SBATCH -N 1 -n 8


source activate agent

python main.py --nimbus_table ./data/nimbus_cell_table.csv --results ./results/results.csv