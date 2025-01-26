# Run

## Method 1

To install all packages:
Run in main folder: `conda env create -f environment.yaml`

## Method 2

You can install all packages directly using below steps:
**Create local conda venv named `pytorch_env` and run** \
```multi-agent-RL>conda init```
```multi-agent-RL>conda activate pytorch_env```
```multi-agent-RL>conda install -c conda-forge pettingzoo stable-baselines3 matplotlib numpy```
```multi-agent-RL>conda install -c conda-forge pettingzoo[mpe]```
```multi-agent-RL>conda install -c conda-forge tensorboard```
```multi-agent-RL>pip install supersuit```
```multi-agent-RL>set PYGAME_DETECT_AVX2=1 && conda install pygame```

# Verification

**To check if RL env is working run** `test.py` script

**To run tensorboard result for specific experiment**:
```multi-agent-RL\src\results>tensorboard --logdir=vdn_mpe_model\0.9_gamma\25000_epochs\0.0001_lr\0.9996_eps\tensorboard_logs```

**On browser run:** `http://localhost:6006/`