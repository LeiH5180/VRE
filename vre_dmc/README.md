This repo is an implementation of VRE

## setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:

conda env create -f setup/conda.yaml
conda activate dmcgb
sh setup/install_envs.sh

## Datasets
Part of this repository relies on external datasets. SODA uses the Places dataset for data augmentation, which can be downloaded by running

wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

Distracting Control Suite uses the DAVIS dataset for video backgrounds, which can be downloaded by running
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

## Use tensorboard to visualize training and testing
tensorboard --logdir=logs/path/to/tb