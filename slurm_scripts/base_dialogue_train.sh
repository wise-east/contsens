#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --time=5:00:00 
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB

CONDA_ENV="contsens"
# source $HOME/miniconda/etc/profile.d/conda.sh
source $HOME/.bashrc
source $PROJ_DIR/csci699_dcnlp_projectcode/slurm_scripts/set_envs.sh
cd $PROJ_DIR/csci699_dcnlp_projectcode

# usage: 
# sbatch base_dialogue_train.sh "<ARGS>"
# . base_dialogue_train.sh "<ARGS>"
# e.g. sbatch base_dialogue_train.sh "$ARGS"
ARGS=$1
contsens $ARGS