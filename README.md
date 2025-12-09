# ESE 3060 Final Project Fall 2025

## PART 1 (CIFAR-10) Instructions + Guide:
Install all required packages:
```bash
pip install -r requirements.txt
```

Then, to run baseline:
```bash
python3 airbench94.py
```

To run with depthwise-separable convolution modification:
```bash
python3 airbench94-DS.py
```

To run with batch_size/LR/fused kernel modification:
```bash
python3 airbench94-BatchLR.py
```

To run with both DS and batch_size/LR/fused kernel modifications:
```bash
python3 airbench94-DS-BatchLR.py
```

Each python script produces:
- A corresponding experiment logs folder in ```expt_logs/```
  - This contains many .jsons corresponding to each run
  - Each log contains information on the random seed, hyperparameters, etc.
- A summary table.csv (located in the corresponding ```*_summary/``` directory
- Training loss, validation acc, and time elapsed plots in the ```plots/``` directory

There is also a python script called
```bash
analytics.py
```

Running
```bash
python3 analytics.py
```

will print to the terminal a series of statistical metrics (mean, std dev, confidence intervals) when provided with a file path name to a summary table csv.

NOTE: Running analytics.py may need additional pip installs; it uses pandas, numpy, and scipy.

ALSO NOTE: We have commit hashes in my expt_logs, but we did not have time to properly link GitHub and Runpod.io, so git commit hashes may be static/outdated from when I first copied the repository into the Runpod instance.

================================================================

## Project Overview
This project contains two machine learning training benchmarks:
- **airbench94.py**: CIFAR-10 image classification benchmark
- **train_gpt.py**: GPT-2 training on the FineWeb-10B dataset

## Setup and Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (A100/H100 recommended)
- CUDA 11.7 or later

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

## Running airbench94.py

### Overview
CIFAR-10 training benchmark achieving 94.01% average accuracy in 3.83 seconds on an NVIDIA A100. You will want to use a single node of an a100.
- CIFAR-10 dataset automatically downloaded on first run
- Cached to `cifar10/` directory as `.pt` files for faster subsequent runs

### Execution
```bash
python airbench94.py
```

Runs 25 training iterations and reports mean/standard deviation accuracy metrics.

### Output
- Per-epoch training metrics (loss, accuracy)
- Validation and test-time augmentation (TTA) accuracy
- Logs saved to `logs/{uuid}/log.pt`

### Hardware Requirements
- NVIDIA A100 GPU recommended
- CUDA 11.7+
- NVIDIA Driver 515.105.01 or compatible

### Reference
Based on: [cifar10-airbench legacy airbench94.py](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py)

## Running train_gpt.py

### Overview
Trains a GPT-2 model on the FineWeb-10B dataset. You will want to use an 8xH100.

### Execution
Download the data with 
```bash
python cached_fineweb10B.py 9
```
and then run the script with 
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Hardware Requirements
- Tested on 8× NVIDIA H100 80GB GPUs
- PyTorch 2.4.1+ with CUDA 12.1

### Reference
Based on: [modded-nanogpt record number #5](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt)
