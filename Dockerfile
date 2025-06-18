FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ARG CONDA_ENV_NAME=pytorch-dev
ARG PYTHON_VERSION=3.10.0

# 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl vim wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# miniconda 설치
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp ${CONDA_DIR} && \
    rm /tmp/miniconda.sh
RUN conda update -y conda

# Create the conda environment
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}
ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH

# Install the packages
COPY requirements.txt /tmp/requirements.txt
RUN conda run -n ${CONDA_ENV_NAME} pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
CMD ["bash"]