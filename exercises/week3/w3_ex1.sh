
#!/bin/bash
#BSUB -J w3_ex1
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "rusage[mem=1024MB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -o batch_outputs/w3_%J.out
#BSUB -e batch_errors/w3_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python -u w3_ex1.py
