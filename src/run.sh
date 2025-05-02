#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem=100G
#SBATCH -J modis_preproc
#SBATCH -o logs/modis_preproc.out
#SBATCH -e logs/modis_preproc.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=__________@brown.edu #<-- whoever's email

# Clear any modules and load required ones
module purge
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

# Create virtual environment if it doesn't exist
if [ ! -d "$HOME/______" ]; then
  echo "Creating virtual environment..."
  python -m venv $HOME/_______
fi

# Activate your virtual environment
source $HOME/_______/bin/activate  # <-- whatever our environment name will be

# Install required packages --> hopefully this is extensive enough
pip list | grep -q numpy || pip install numpy
pip list | grep -q tensorflow || pip install tensorflow
pip list | grep -q scikit-learn || pip install scikit-learn
pip list | grep -q pyhdf || pip install pyhdf
pip list | grep -q zarr || pip install zarr
pip list | grep -q matplotlib || pip install matplotlib
pip list | grep -q pandas || pip install pandas
pip list | grep -q tqdm || pip install tqdm



# Create logs directory if it doesn't exist
mkdir -p logs

# Find the actual CUDA path based on the loaded module
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
echo "Found CUDA at: $CUDA_PATH"

# Set environment variables
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Print information about the GPU
echo "CUDA Devices:"
nvidia-smi

# Create the data directories if they don't exist
mkdir -p data #Where we can store the data 

# Run the preprocessing script
echo "Starting data preprocessing..."
python -u ______.py #whatever the python file is called 
