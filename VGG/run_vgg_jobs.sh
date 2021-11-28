#!/bin/bash -l
sbatch vgg16_dense.sh
sbatch vgg16_gaussiank.sh
sbatch vgg16_gtopk.sh
sbatch vgg16_oktopk.sh
sbatch vgg16_topkA.sh
sbatch vgg16_topkDSA.sh
