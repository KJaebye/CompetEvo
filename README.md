# CompetEvo

Morph Evolution and Strategies Learning via Competition.

This work is modefied based on awesome works **openai/multiagent-competition** https://github.com/openai/multiagent-competition.git, and **inspirai/TimeChamber** https://github.com/inspirai/TimeChamber.git. The former one built the competition environments for the paper <a href="https://arxiv.org/abs/1710.03748">Emergent Complexity via Multi-agent Competition</a>, and the latter re-implemented these on IsaacGym preview release 4 for large-scale training possibility.

## Installation
Similiar with <a href="https://github.com/inspirai/TimeChamber/tree/main#installation">TimeChamber</a>. <a href="https://developer.nvidia.com/isaac-gym">**IsaacGym preview release 4**</a> is required. Then install this repo:
```bash
pip install -e .
```
> Note: IsaacGym is currently a module of NVIDIA Omniverse, and the preview versions are not maintained anymore. Some old APIs have been changed in new Omniverse IsaacGymEnvs. So this repo only works on **IsaacGym Preview Release 4**.

## Training


## Troubleshooting
### CUDA incompatibility
IsaacGym Preview Release4 official python environment is built on `torch==1.8`, `cudatoolkit==11.1`, which is quit old for the latest GPU hardware like 4090. Error occours:
```
NVIDIA GeForce RTX 4090 with CUDA capability sm_89 is not compatible with the current PyTorch installation.
```
Install the latest pytorch release with `CUDA>=11.7` can solve this.