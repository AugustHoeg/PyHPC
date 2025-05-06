
#!/bin/bash
#BSUB -J w4_ex8
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o batch_outputs/w4_%J.out
#BSUB -e batch_errors/w4_%J.err

lscpu

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python -u w4_ex8.py /dtu/projects/02613_2025/data/locations/locations_100.csv

