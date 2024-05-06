#!/bin/bash
#SBATCH --nodes=2          
#SBATCH -p gpu --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  
#SBATCH --mem-per-cpu=16G   
#SBATCH -t 10:00:00
#SBATCH --mail-type=begin     
#SBATCH --mail-type=end       
#SBATCH --mail-user=ethan_williams@brown.edu

module purge
unset LD_LIBRARY_PATH
srun apptainer exec --nv tensorflow-24.03-tf2-py3.simg python3 src/main.py $@

# bash src/experiment.sh --mode experiment --run-id hello --exclude noise --learning-rate 1e-7
