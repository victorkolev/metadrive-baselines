from tensorflow/tensorflow:2.6.0-gpu

RUN \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8-venv \ 
    python3.8-dev \
    python3-setuptools \
    python3-opencv \
    xvfb \
    curl vim git make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /moto

COPY . .
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

EXPOSE 80

RUN pip install --no-cache-dir wheel==0.38.4
RUN pip install --no-cache-dir -U -r requirements.txt
RUN pip install --no-cache-dir "metadrive @ git+https://github.com/metadriverse/metadrive"
RUN pip install --no-cache-dir panda3d-gltf==0.13


