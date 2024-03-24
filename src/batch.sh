#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=4G

module load python/3.11.0s-ixrhc3q

make "$1"
