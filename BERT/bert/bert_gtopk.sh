#!/bin/bash -l
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --output=32nodes_bert_gtopk_density1.txt


module load daint-gpu
module load cudatoolkit/10.2.89_3.28-2.1__g52c0314

conda activate py38_oktopk
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
        --num_minibatches 256 \
	--density 0.01 \
	--compressor 'gtopk' \
        --gradient_accumulation_steps 1 --dataparallel --config_path tests/depth=4/conf_32nodes.json


