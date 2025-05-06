
#!/bin/bash
#BSUB -J w2_ex5
#BSUB -q hpc
#BSUB -W 2
#BSUB -R rusage[mem=512MB]
#BSUB -o batch_outputs/w2_ex5_%J.out
#BSUB -e batch_errors/w2_ex5_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python -u w2_ex5.py 1 2 3 4 5 6 7 8 9 10
