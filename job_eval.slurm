#! /bin/tcsh

#SBATCH --job-name=awesome-conformer-eval
#SBATCH --output=results/eval.out # redirect stdout
#SBATCH --error=results/eval.err # redirect stderr
#SBATCH --partition=studentbatch # (Sharyn's best partition yet)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=8 # GPUs in total

alias conda2 'source /home/yandex/APDL2425a/group_8/bin/Anaconda3/etc/profile.d/conda.csh'
conda2 activate conformer0
setenv HF_HOME /home/yandex/APDL2425a/group_8/bin/HuggingFace-cache
python train.py hparams/conformer_small.yaml --data_folder datasets/LibriSpeech/
