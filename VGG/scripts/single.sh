#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=dnn_single_node_1.txt


module load daint-gpu
module load PyTorch


which nvcc
nvidia-smi

which python



dnn="${dnn:-resnet20}"
#dnn="${dnn:-resnet50}"
echo $dnn
source exp_configs/$dnn.conf
nstepsupdate=1
srun python dl_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate
