#ARG BASE_TAG
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
software-properties-common \
wget \
net-tools \
curl \
sudo \
automake \
perl \
git-core
#wget \
#&& apt-get clean \
#&& rm -rf /var/lib/apt/lists/*

# Set Locale
RUN apt-get install -y language-pack-ko
#RUN apt-get install -y language-pack-ko \
#&& apt-get clean \
#&& rm -rf /var/lib/apt/lists/*

RUN locale-gen ko_KR.UTF-8

# Install SWIG
RUN apt-get install -y swig

# Install vim
RUN apt-get install -y vim
#RUN apt-get install -y vim \
#&& apt-get clean \
#&& rm -rf /var/lib/apt/lists/*

# Install python3
RUN add-apt-repository -y ppa:jonathonf/python-3.6 \
&& apt-get update && apt-get install -y python3.6 python3.6-dev \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* \
\
&& ln -sf /usr/bin/pydoc3.6 /usr/bin/pydoc3 \
&& ln -sf /usr/bin/python3.6 /usr/bin/python3 \
&& ln -s /usr/bin/python3.6-config /usr/bin/python3-config \
&& ln -s /usr/bin/pydoc3 /usr/bin/pydoc \
&& ln -s /usr/bin/python3 /usr/bin/python \
&& ln -s /usr/bin/python3-config /usr/bin/python-config

# Install pip
RUN wget -q -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
&& python /tmp/get-pip.py \
&& rm /tmp/get-pip.py

# Install pytorch
RUN pip3 install torch torchvision

# Install AllenNLP
RUN pip install allennlp==0.8.0

# Install elasticsearch
RUN pip3 install elasticsearch==6.3.1

ENV LANG ko_KR.UTF-8

# Install apache
RUN apt-get update
RUN apt-get -y install apache2

# Config apache to use cuda
RUN cd /usr/lib \
&& ln -s /usr/local/cuda/lib64/* ./

# Install python module for deployment
RUN pip install imutils requests python-crfsuite

# Install logrotate
RUN apt-get install -y logrotate

# Install pytorch-BERT
RUN pip install pytorch_pretrained_bert

# Install Mecab
RUN wget -q -O /tmp/mecab-0.996-ko-0.9.2.tar.gz https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz \
&& cd /tmp \
&& tar xvfz mecab-0.996-ko-0.9.2.tar.gz \
&& cd mecab-0.996-ko-0.9.2 \
&& ./configure \
&& make \
&& make install \
&& ldconfig \
&& cd /tmp \
&& rm -f mecab-0.996-ko-0.9.2.tar.gz

# Install Mecab dict
RUN wget -q -O /tmp/mecab-ko-dic-2.1.1-20180720.tar.gz https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz \
&& cd /tmp \
&& tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz \
&& cd mecab-ko-dic-2.1.1-20180720 \
&& ./autogen.sh \
&& ./configure \
&& make \
&& make install \
&& ldconfig \
&& cd /tmp \
&& rm -f mecab-ko-dic-2.1.1-20180720.tar.gz

# Binding Mecab to python
RUN git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git \
&& cd mecab-python-0.996 \
&& python setup.py build \
&& python setup.py install

# Clean apt-get
RUN apt-get clean \
&& rm -rf /var/lib/apt/lists/* 
