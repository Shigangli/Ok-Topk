## Near-Optimal Sparse Allreduce for Distributed Deep Learning (PPoPP'22)
Ok-Topk is a scheme for distributed training with sparse gradients. Ok-Topk integrates a novel sparse allreduce algorithm (less than 6k communication volume which is asymptotically optimal) with the decentralized parallel Stochastic Gradient Descent (SGD) optimizer, and its convergence is proved theoretically and empirically.

## Prepare Datasets

### Cifar-10 for VGG
`cd ./VGG/vgg_data`

`wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`

`tar -zxvf cifar-10-python.tar.gz`

### AN4 for LSTM
`cd ./LSTM/audio_data`

`wget https://www.dropbox.com/s/l5w4up20u5pfjxf/an4.zip`

`unzip an4.zip`

### Wikipedia for BERT
`cd ./BERT/bert/bert_data/`

Prepare the dataset according to the README file.

## Setup the environment
To install the required Python modules: 

`conda create --name py38_oktopk python=3.8`

`conda activate py38_oktopk`

`pip install -r requirements.txt`

## Run jobs
We run experiments on GPU clusters with SLURM job scheduler.
To evaluate the performance of `Ok-Topk`, `Gaussiank`, `gtopk`, `topkA`, `topkDSA`, and `dense`, run the jobs as follows.

### To run VGG jobs
`cd ./VGG`

`./sbatch_vgg_jobs.sh`

### To run LSTM jobs
`cd ./LSTM`

`./sbatch_lstm_jobs.sh`

### To run BERT jobs
`cd ./BERT/bert/`

`./sbatch_bert_jobs.sh`

## License

See [LICENSE](LICENSE).
