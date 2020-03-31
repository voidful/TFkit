FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

ENV LANG=C.UTF-8
WORKDIR /workspace/
COPY ./ /workspace/

# install basics
RUN apt-get update -y
RUN apt-get install -y git curl htop wget tmux

# install python deps
RUN pip install -r /workspace/requirements.txt
