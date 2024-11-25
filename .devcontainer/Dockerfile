FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# set the working directory in the container
WORKDIR /workspaces/foundation-graph-segmentation

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# install python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip python3.10-dev python3.10-venv && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# install pytorch cuda
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

USER user
