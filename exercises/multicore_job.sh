
#!/bin/bash

#BSUB -q hpc
#BSUB -J August
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- Select the resources: 1 gpu in exclusive process mode --
###BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 40GB of system-memory rusage=40
###BSUB -R "select[gpu40gb]"
#BSUB -R "rusage[mem=512MB]"
#BSUB -u "august.hoeg@gmail.com"
#BSUB -B
#BSUB -N
#BSUB -oo batch_outputs/output_august_%J.out
#BSUB -eo batch_errors/error_august_%J.out

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python -u script.py
