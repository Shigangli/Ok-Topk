#!/bin/bash -l
sbatch bert_dense.sh
sbatch bert_gaussiank.sh
sbatch bert_gtopk.sh
sbatch bert_oktopk.sh
sbatch bert_topkA.sh
sbatch bert_topkDSA.sh
