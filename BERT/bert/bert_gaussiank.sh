#!/bin/bash -l
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --account=g34
#SBATCH --output=32nodes_bert_gaussiank_density1.txt


module load daint-gpu
module load cudatoolkit/10.2.89_3.29-7.0.2.1_3.27__g67354b4


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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
echo $MASTER_ADDR

srun python -m mpi4py main_bert.py \
        --module models.bert12.depth=4 \
        --max_seq_length 128 \
        --train_batch_size 8 \
        --train_path ./bert_data/wikipedia.segmented.nltk.txt \
        --bert_config_path configs/bert_config_bert-base-uncased.json \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --do_train \
        --do_lower_case \
        --num_minibatches 512 \
	--density 0.01 \
	--compressor 'gaussiank' \
        --gradient_accumulation_steps 1 --dataparallel --config_path tests/depth=4/conf_32nodes.json

