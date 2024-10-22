# CompetEvo
This repo provides code for IJCAI-2024 paper "CompetEvo: Towards Morphological Evolution from Competition". 

Abstract: Training an agent to adapt to specific tasks through co-optimization of morphology and control has gradually attracted attention. However, whether there exists an optimal configuration and tactics for agents in a multiagent competition scenario is still an issue that is challenging to definitively conclude. In this context, we propose competitive evolution (CompetEvo), which co-evolves agents' designs and tactics in confrontation. We build arenas consisting of three animals and their evolved derivatives, placing agents with different morphologies in direct competition with each other. The results reveal that our method enables agents to evolve a more suitable design and strategy for fighting compared to fixed-morph agents, allowing them to obtain advantages in combat scenarios. Moreover, we demonstrate the amazing and impressive behaviors that emerge when confrontations are conducted under asymmetrical morphs.

See more details on https://competevo.github.io/.

# Build Image
```
cd docker
docker build -t 'kjaebye/competevo:1.0' .   
```

# Run Container
```
xhost +
docker run -it --name competevo -v ~/ws:/root/ws --gpus=all <enter the IMAGE ID> /bin/bash
```
Download this repo into ~/ws on the host, and the docker volume could map the `~/ws` to `/root/ws` in containner:
```
cd ~/ws
git clone git@github.com:KJaebye/CompetEvo.git
```

# To test the environment is working
We give a pretrained model in task robo-sumo-devants-v0.
```
python display.py --cfg config/robo-sumo-devants-v0.yaml --ckpt_dir runs/robo-sumo-devants-v0/models
```
You can see a pair of developed ants fighting on an arena.

# Training
```
python train.py --cfg config/run-to-goal-ants-v0.yaml
```
cfg files that can be selected to train: 

- config/run-to-goal-ants-v0.yaml
- config/run-to-goal-bugs-v0.yaml
- config/run-to-goal-spiders-v0.yaml
- config/run-to-goal-devants-v0.yaml
- config/run-to-goal-devbugs-v0.yaml
- config/run-to-goal-devspiders-v0.yaml
- config/robo-sumo-ants-v0.yaml
- config/robo-sumo-bugs-v0.yaml
- config/robo-sumo-spiders-v0.yaml
- config/robo-sumo-devants-v0.yaml
- config/robo-sumo-devbugs-v0.yaml
- config/robo-sumo-devspiders-v0.yaml


# Dispaly
```
python display.py --cfg <path-to-config-file> --ckpt_dir <path-to-models-directory>
```
