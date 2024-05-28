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
docker run -it --name competevo -v ws:/root/ws --gpus=all -v /tmp/.x11-unix:/tmp/.x11-unix -e GDK_SCALE -e GDK_DPI_SCALE -p 8022:22 11aa1b03c99f /bin/bash
```

# Training
```
python train.py --cfg config/run-to-goal-ants-v0.yaml
```

# Dispaly
```
python display.py --run_dir ...
```
