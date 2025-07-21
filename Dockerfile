FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS base

FROM base AS base-amd64

ENV NV_CUDNN_VERSION=9.5.1.17-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-12=${NV_CUDNN_VERSION}

FROM base AS base-arm64

ENV NV_CUDNN_VERSION=9.5.1.17-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-12=${NV_CUDNN_VERSION}
FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# 下载并编译安装 Python 3.12
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tar.xz && \
    tar -xf Python-3.12.0.tar.xz && \
    cd Python-3.12.0 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf Python-3.12.0 Python-3.12.0.tar.xz

# 创建 python 和 pip 的符号链接
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip3

# 更新 pip 并验证安装
RUN python -m pip install --upgrade pip\
    && python --version \
    && pip --version

# 3. 验证安装
RUN python3.12 --version && \
    /usr/local/bin/pip --version

# 3. 设置工作目录
WORKDIR /app

# 4. 复制文件
COPY requirements.txt /app/

COPY third_parties/ /app/third_parties/

# 4. 安装 Python 依赖
# ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple



# thundersvm
# cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..
# cd ../python
# python setup.py install
# sudo apt-get install libnuma-dev

# 安装 flash-attn


COPY ./code/ /app/code/
# COPY ./model/ /app/model/
COPY ./init.sh /app/init.sh
COPY ./train.sh /app/train.sh
COPY ./test.sh /app/test.sh
COPY ./readme.pdf /app/readme.pdf

EXPOSE 8888
ENV JUPYTER_WORKSPACE_NAME=app
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--port=8888"]
# CMD ["ipython"]

