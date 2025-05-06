
#!/bin/bash
#BSUB -J sleeper
#BSUB -q hpc
#BSUB -W 2
#BSUB -R rusage[mem=512MB]
#BSUB -o batch_outputs/sleeper_%J.out
#BSUB -e batch_errors/sleeper_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python -u script.py
