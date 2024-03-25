#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=4G
#SBATCH --partition=batch

module load python

make "$1"
