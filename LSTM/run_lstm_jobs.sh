#!/bin/bash -l
sbatch lstm_dense.sh
sbatch lstm_gaussiank.sh
sbatch lstm_gtopk.sh
sbatch lstm_oktopk.sh
sbatch lstm_topkA.sh
sbatch lstm_topkDSA.sh
