FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 
# FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04 

MAINTAINER Tabish Rashid

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# COPY sources.list  /etc/apt/sources.list 
# RUN sed -i '/nvidia/d' /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget tmux sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame -i https://pypi.douban.com/simple/

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6 -i https://pypi.douban.com/simple/
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools -i https://pypi.douban.com/simple/
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch==1.4.0+cu92  torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.douban.com/simple/
# RUN pip3 install torchvision==0.5.0+cu92  snakeviz pytest probscale -i https://pypi.douban.com/simple/

## -- SMAC
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /starcraft/pymarl/3rdparty/StarCraftII

## -- Yuxuan 
RUN pip install tensorboardX -i https://pypi.douban.com/simple/

# WORKDIR /pymarl
# COPY ../../pymarl/3rdparty /pymarl
# ADD ../../pymarl/3rdparty/ /pymarl/3rdparty

