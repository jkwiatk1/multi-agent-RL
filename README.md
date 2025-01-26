# Multi-Agent Reinforcement Learning (MARL) - Qatten/DQN/VDN

## Overview

This project implements multi-agent reinforcement learning (MARL) algorithms to solve environments in the **PettingZoo**
multi-agent benchmark suite.
The focus is on algorithms

* Qatten
* VDN
* DQN

The repository provides training scripts, environment configurations, and tools for analyzing results using.

---

## Installation Guide

### Method 1: Using `environment.yaml` (Recommended)

To install all required packages and dependencies with Conda:

1. Clone this repository:
   ```bash
   git clone https://github.com/jkwiatk1/multi-agent-RL
   cd multi-agent-RL
2. Create a Conda environment:
   ```bash
   conda env create -f environment.yaml
   ```
3. Activate the environment:
    ```bash
    conda activate pytorch_env
    ```

### Method 2: Manual Installation

Follow these steps to manually install the required dependencies:

1. Initialize Conda and check existing environments:
    ```bash
    conda init
    conda env list

2. Create a new Conda environment named `pytorch_env` and activate it:
    ```bash
   conda create --name pytorch_env python=3.9 -y
   conda activate pytorch_env
3. Install the required packages:
   ```bash
   conda install -c conda-forge pettingzoo stable-baselines3 matplotlib numpy
   conda install -c conda-forge pettingzoo[mpe]
   conda install -c conda-forge tensorboard
   pip install supersuit
   set PYGAME_DETECT_AVX2=1 && conda install pygame

## Verification

To ensure the environment is working:

* Run the `test.py` script

## Experiments

1. To run experiments open **src.training.experiments.py** script.
2. Configure params dictionaries
3. Run script
    1. Results of training will be store under specific path.

## Evalution

1. To evalute the trained model open **src.evalution** folder
2. Open one of the script:
    1. for DQN models: `evaluate_dqn.py`
    2. for VDN models: `evaluate_vdn.py`
    3. for Qatten model: `evaluate_qatten.py`
3. Configure path where the results of trained model (train weights are stored there)
4. Run script
5. Results will be stored there.

## Viewing Results in TensorBoard

To visualize experiment results using TensorBoard. Run example:

* ```
  multi-agent-RL\src\results>tensorboard --logdir=vdn_mpe_model\0.9_gamma\25000_epochs\0.0001_lr\0.9996_eps\tensorboard_logs
  ```

Open TensorBoard in your browser by navigating to:

* `http://localhost:6006/`