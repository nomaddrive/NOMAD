# Base image with CUDA and cuDNN support                                                                                      
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y -q --no-install-recommends \
	software-properties-common \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        wget \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        mesa-common-dev \
        libc++1 \
        openssh-client \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11
RUN apt-add-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y -q --no-install-recommends python3.11 python3.11-dev python3.11-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 11

RUN apt-get remove -y cmake && pip3 install  --no-cache-dir --upgrade cmake

# NOTE: the project-shipped gpudrive native engine is built further down,
# after the Python deps, so the heavy torch/TF/JAX layers stay cached.

RUN pip3 install --no-cache-dir torch==2.6.0 && rm -rf ~/.cache/pip/*
RUN pip3 install --no-cache-dir tensorflow==2.19.0 && rm -rf ~/.cache/pip/*
RUN pip3 install --no-cache-dir stable-baselines3[extra] && rm -rf ~/.cache/pip/*
RUN pip3 install --no-cache-dir git+https://github.com/PufferAI/PufferLib.git@gpudrive  && rm -rf ~/.cache/pip/*
RUN pip3 install --no-cache-dir nvidia-cuda-runtime-cu12==12.4.127 && rm -rf ~/.cache/pip/*
RUN pip3 install --no-cache-dir 'jax[cuda12]<=0.6.0' && rm -rf ~/.cache/pip/*

# Install the gpudrive package runtime dependencies. These come from
# pyproject.toml [project].dependencies, plus scipy/scikit-learn/shapely/
# imageio/rich which the baselines import directly but pyproject omits.
RUN pip3 install --no-cache-dir \
      "numpy>=1.26.4,<2" gymnasium pygame "matplotlib==3.9" pandas \
      "python-box==7.2.0" typer pyyaml mediapy wandb seaborn safetensors \
      tqdm huggingface_hub scipy scikit-learn shapely imageio rich \
    && rm -rf ~/.cache/pip/*


# ---- Build the project-shipped (NOMAD) gpudrive native engine ----
# This repo declares submodules in .gitmodules but records no commit pins,
# so restore them explicitly from the declared sources:
#   external/madrona -> m-naumann/madrona @ 4bda334
#       (branch gpudrive/compatible_build_backend_naming; identical commit to
#        upstream Emerge-Lab/gpudrive's madrona pin)
#   external/json    -> nlohmann/json @ 0457de2 (upstream pin; same URL)
COPY . /gpudrive
WORKDIR /gpudrive
RUN rm -rf external && mkdir -p external \
    && git clone https://github.com/m-naumann/madrona.git external/madrona \
    && git -C external/madrona checkout 4bda33465340fabc2e61fb27f95aa04795a15466 \
    && git -C external/madrona submodule update --init --recursive \
    && git clone https://github.com/nlohmann/json.git external/json \
    && git -C external/json checkout 0457de21cffb298c22b629e538036bfeb96130b7 \
    && printf 'set(MADRONA_ENABLE_TESTS ON)\nadd_subdirectory("${MADRONA_DIR}" madrona EXCLUDE_FROM_ALL)\nadd_subdirectory(json)\n' > external/CMakeLists.txt

RUN mkdir -p build
WORKDIR /gpudrive/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && find external -type f -name "*.tar" -delete
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH make -j"$(nproc)" \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1
WORKDIR /gpudrive
# Make the gpudrive package + compiled madrona_gpudrive extension importable
ENV PYTHONPATH=/gpudrive:/gpudrive/build


RUN groupadd -g 1014 zilin &&\
    useradd -l -u 3725 -g zilin zilin &&\
    install -d -m 0755 -o zilin -g zilin /home/zilin &&\
    chown --changes --silent --no-dereference --recursive 3725:1014 /home/zilin

ENV MADRONA_MWGPU_KERNEL_CACHE=/home/zilin/gpudrive_cache

USER zilin
WORKDIR /workspace

CMD ["/bin/bash"]
LABEL org.opencontainers.image.source=https://github.com/Emerge-Lab/gpudrive
