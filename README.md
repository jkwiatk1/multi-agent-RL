**Create local conda venv named `pytorch_env` and run** \
```USD_24Z\multi-agent-RL>conda init```
```USD_24Z\multi-agent-RL>conda env list```
```USD_24Z\multi-agent-RL>conda activate pytorch_env```
```USD_24Z\multi-agent-RL>conda install -c conda-forge pettingzoo stable-baselines3 matplotlib numpy```
```USD_24Z\multi-agent-RL>conda install -c conda-forge pettingzoo[mpe]```
```USD_24Z\multi-agent-RL>conda install -c conda-forge tensorboard```
```USD_24Z\multi-agent-RL>pip install supersuit```
```USD_24Z\multi-agent-RL>set PYGAME_DETECT_AVX2=1 && conda install pygame```

**To check if RL env is working run** `test.py` script

**To run tensorboard result** example:
```USD_24Z\multi-agent-RL\src\results>tensorboard --logdir=vdn_mpe_model\20000_epochs\0.0001_lr\0.9996_eps\tensorboard_logs```

On browser run: `http://localhost:6006/`