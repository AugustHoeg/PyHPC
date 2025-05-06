
#!/bin/bash
#BSUB -J w3_ex5
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o batch_outputs/w3_%J.out
#BSUB -e batch_errors/w3_%J.err

lscpu

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#python -u w3_ex5.py

perf stat -e L1-dcache-load-misses, LLC-load-misses python w3_ex5.py
