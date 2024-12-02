# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=20.04
ARG NVIDIA_CUDA_VERSION=11.3.1
# CUDA architectures, required by Colmap and tiny-cuda-nn. Use >= 8.0 for faster TCNN.
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"

# Pull source either provided or from git.
FROM scratch as source_copy

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder
ARG CUDA_ARCHITECTURES
ARG NVIDIA_CUDA_VERSION
ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV QT_XCB_GL_INTEGRATION=xcb_egl

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y 'ppa:deadsnakes/ppa' && \
    apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libpython3.8-dev \
        python3.8-dev \
        python3-dev \
        python3-pip

# Upgrade pip and install dependencies.
# pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118 && \
RUN pip install --no-cache-dir --upgrade pip 'setuptools<70.0.0' && \
    pip install --no-cache-dir torch==1.12.0+cu113 torchvision==0.13.0+cu113 'numpy<2.0.0' --extra-index-url https://download.pytorch.org/whl/cu113
    
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export FORCE_CUDA=1 && \
    pip install --no-cache-dir open-clip-torch==2.0.2  transformers==4.18.0 flash-attn==0.2 xformers==0.0.13 motion-vector-extractor==1.0.6 && \
    pip install --no-cache-dir simplejson pynvml easydict fairscale oss2 scikit-video scikit-image imageio ipdb rotary-embedding-torch==0.2.1 && \
    pip install --no-cache-dir pytorch-lightning==1.4.2 torchmetrics==0.6.0

        
# Fix permissions
RUN chmod -R go=u /usr/local/lib/python3.8

#
# Docker runtime stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
ARG CUDA_ARCHITECTURES
ARG NVIDIA_CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"

# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y 'ppa:deadsnakes/ppa' && \
    apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        python3.8 \
        python3.8-dev \
        build-essential \
        python-is-python3 \
        ffmpeg

# Copy packages from builder stage.
COPY --from=builder /usr/local/lib/python3.8/dist-packages/ /usr/local/lib/python3.8/dist-packages/

# Bash as default entrypoint.
CMD /bin/bash -l
