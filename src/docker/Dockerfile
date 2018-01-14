# Builds stanfordsquad/ubuntu:1.3

FROM ubuntu:14.04
MAINTAINER Konstantin Lopyrev <klopyrev@gmail.com>

RUN mkdir -m 777 /user

RUN apt-get update -y

RUN apt-get install -y python-pip python-dev

# Install Tensorflow.
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# Set up protocol buffers.
RUN apt-get install -y wget
RUN wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-python-3.0.0-beta-2.tar.gz && tar xzf protobuf-python-3.0.0-beta-2.tar.gz
RUN cd protobuf-3.0.0-beta-2 && ./configure && make -j 8 && make install
RUN rm -rf protobuf-3.0.0-beta-2 protobuf-python-3.0.0-beta-2.tar.gz
RUN ldconfig
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp27-none-linux_x86_64.whl

# Get add-apt-repository.
RUN apt-get install -y software-properties-common python-software-properties

# Java 1.8 (used for CoreNLP)
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get -y update
RUN apt-get -y install openjdk-8-jdk

# Set up the source directory.
RUN printf "\nexport PYTHONPATH=src:$PYTHONPATH\n" >> /user/.bashrc

# Google Flags
RUN apt-get install -y libgflags-dev

# Boost
RUN apt-get install -y libboost-dev libboost-system-dev libboost-thread-dev

WORKDIR /user
CMD /bin/bash
