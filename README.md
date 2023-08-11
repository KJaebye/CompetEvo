# EMAT-mujoco

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