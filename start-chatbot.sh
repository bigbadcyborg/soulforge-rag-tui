#!/usr/bin/env bash

cd /mnt/c/Users/sully/Documents/projects/chatbot-uncensored || exit 1

export CUDA_HOME=/usr/local/cuda-12.9
export CUDAToolkit_ROOT=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

source .venv-wsl/bin/activate

python chatbot.py

exec bash