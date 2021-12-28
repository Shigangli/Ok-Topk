#!/bin/bash -l
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --output=16nodes_vgg_gaussiank_density2.txt


module load daint-gpu
conda activate py38_oktopk
which nvcc
nvidia-smi

which python

dnn="${dnn:-vgg16}"
density="${density:-0.02}"
source exp_configs/$dnn.conf
compressor="${compressor:-gaussiank}"
nworkers="${nworkers:-16}"

echo $nworkers
nwpernode=1
sigmascale=2.5
PY=python

srun $PY -m mpi4py main_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nwpernode $nwpernode --nsteps-update $nstepsupdate --compression --sigma-scale $sigmascale --density $density --compressor $compressor
