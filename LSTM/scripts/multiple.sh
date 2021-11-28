#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=dnn_2nodes_2dense.txt


module load daint-gpu
#module load PyTorch


__conda_setup="$('/project/g34/shigang/anaconda38/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/project/g34/shigang/anaconda38/etc/profile.d/conda.sh" ]; then
        . "/project/g34/shigang/anaconda38/etc/profile.d/conda.sh"
    else
        export PATH="/project/g34/shigang/anaconda38/bin:$PATH"
    fi
fi
unset __conda_setup
which nvcc
nvidia-smi

which python



dnn="${dnn:-resnet20}"
density="${density:-0.001}"
source exp_configs/$dnn.conf
compressor="${compressor:-gtopk}"
nworkers="${nworkers:-2}"

echo $nworkers
nwpernode=1
sigmascale=2.5
PY=python

#srun $PY -m mpi4py gtopk_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nwpernode $nwpernode --nsteps-update $nstepsupdate --compression --sigma-scale $sigmascale --density $density --compressor $compressor
srun $PY -m mpi4py gtopk_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nwpernode $nwpernode --nsteps-update $nstepsupdate --sigma-scale $sigmascale --density $density
