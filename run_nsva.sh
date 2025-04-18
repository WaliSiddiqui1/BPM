#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem=100G
#SBATCH -J nsva_dataset_build
#SBATCH -o logs/nsva_dataset_build.out
#SBATCH -e logs/nsva_dataset_build.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@brown.edu  # <-- replace this with your Brown email

# Clear any modules and load required ones
module purge
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

# Create virtual environment if it doesn't exist
VENV_PATH="$HOME/nsva_env"
if [ ! -d "$VENV_PATH" ]; then
  echo "Creating virtual environment..."
  python3 -m venv $VENV_PATH
fi

# Activate your virtual environment
source $VENV_PATH/bin/activate

# Install required packages (only installs if not present)
pip list | grep -q numpy || pip install numpy
pip list | grep -q opencv-python || pip install opencv-python
pip list | grep -q yt-dlp || pip install yt-dlp
pip list | grep -q tqdm || pip install tqdm
pip list | grep -q torch || pip install torch
pip list | grep -q torchvision || pip install torchvision
pip list | grep -q webvtt-py || pip install webvtt-py
pip list | grep -q ffmpeg-python || pip install ffmpeg-python

# Create logs and data directories if they don't exist
mkdir -p logs
mkdir -p data/videos
mkdir -p data/clips
mkdir -p data/frames

# Find the actual CUDA path based on the loaded module
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
echo "Found CUDA at: $CUDA_PATH"

# Set environment variables
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Print information about the GPU
echo "CUDA Devices:"
nvidia-smi

# Run the dataset preprocessing script
echo "Starting NSVA dataset preprocessing..."
python -u nsva_scraper.py  # <-- replace with the name of your Python script