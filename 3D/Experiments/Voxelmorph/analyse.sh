#!/bin/bash
#SBATCH --gres=gpu:p100:1       # Request GPU "generic resources"
#SBATCH --mem=24G    
#SBATCH --time=00-00:30     # DD-HH:MM:SS


module load python/3.7.7 cuda cudnn

SOURCEDIR=~/scratch/workshop_reg/workshop_registration

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.

## Install required packages
pip install --no-index --upgrade pip
pip install --no-index -r $SOURCEDIR/requirements.txt


# Prepare data
mkdir $SLURM_TMPDIR/dataset
tar xf ~/projects/def-nilanjan/soumyade/neurite.tar -C $SLURM_TMPDIR/dataset



expnum='3'
slurm_data_path=$SLURM_TMPDIR/dataset


python analyse.py   --expnum $expnum \
                    --slurm_data_path $slurm_data_path \
                
 


