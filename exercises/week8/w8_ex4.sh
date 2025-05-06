
#!/bin/bash
#BSUB -J w8_ex4
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o ../batch_outputs/w8_%J.out
#BSUB -e ../batch_errors/w8_%J.err

#lscpu

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

/usr/bin/time -f"mem=%M KB runtime=%e s" 2>&1 python w8_ex4.py


#python -u w6_ex3.py /dtu/projects/02613_2025/data/celeba/celeba_100K.npy
#numactl --interleave=all python -u w6_ex3.py /dtu/projects/02613_2025/data/celeba/celeba_100K.npy