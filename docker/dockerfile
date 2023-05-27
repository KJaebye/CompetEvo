# @ description: Dockerfile for EMAT (Embodied Multi-Agent Task)
# @ author: Kangyao Huang
# @ date: Apirl.05.2023

# If you want to use a different version of CUDA, view the available
# images here: https://hub.docker.com/r/nvidia/cuda
# Note:
#   - Jax currently supports CUDA versions up to 11.3.
#   - Tensorflow required CUDA versions after 11.2.
ARG cuda_docker_tag="11.2.2-cudnn8-devel-ubuntu20.04"
FROM nvidia/cuda:${cuda_docker_tag}

# change apt tsinghua sources
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
COPY . .
ADD sources.list /etc/apt/
RUN apt-get update

# tzdata is required below. To avoid hanging, install it first.
RUN DEBIAN_FRONTEND="noninteractive" apt-get install tzdata -y
RUN apt-get install git wget libgl1-mesa-glx -y

# Install python3.8.
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install python3.8 -y

# Make python3.8 the default python.
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt-get install python3-distutils -y

# Install pip.
RUN apt install python3-pip -y
RUN python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

# Prerequisites
RUN apt-get install \
  libosmesa6-dev \
  libgl1-mesa-glx \
  libglfw3 \
  libglew-dev \
  patchelf \
  gcc \
  python3.8-dev \
  unzip -y \
  libxrandr2 \
  libxinerama1 \
  libxcursor1 \
  vim \
  openssh-server \
  swig

# SSH config
RUN echo "root:123123" | chpasswd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'X11Forwarding yes' >> /etc/ssh/sshd_config
RUN echo 'X11Displayoffset 10' >> /etc/ssh/sshd_config
RUN echo 'X11UseLocalhost no' >> /etc/ssh/sshd_config
RUN service ssh restart

# set SSH auto-on
RUN touch /root/start_ssh.sh
RUN echo '#!/bin/bash \n\
  LOGTIME=$(date "+%Y-%m-%d %H:%M:%S") \n\
  echo "[$LOGTIME] startup run..." >>/root/start_ssh.log \n\
  service ssh start >>/root/start_ssh.log' >> /root/start_ssh.sh
RUN chmod +x /root/start_ssh.sh
RUN echo '# startup run \n\
  if [ -f /root/start_ssh.sh ]; then \n\
      /root/start_ssh.sh \n\
  fi' >> /root/.bashrc

# install Torch
RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# EMAT python env.
WORKDIR /root/
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/