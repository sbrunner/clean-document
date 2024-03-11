# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base
# https://repo.download.nvidia.com/jetson/
# Jetpack 4.6.4
# L4T 32.7.4
# jetson_release

FROM nvcr.io/nvidia/l4t-base:r36.2.0

SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

ENV PATH=/pyenv/shims:/pyenv/bin:${PATH} \
    PYENV_ROOT=/pyenv

# Install different Python version with pyenv
RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    DEV_PACKAGES="build-essential libffi-dev libssl-dev liblzma-dev libsqlite3-dev libcurses-ocaml-dev libreadline-dev libbz2-dev zlib1g-dev" \
    && apt-get update && apt-get install --assume-yes --no-install-recommends git curl zlib1g ${DEV_PACKAGES} \
    && git clone --depth=1 https://github.com/pyenv/pyenv.git /pyenv \
    && pyenv install 3.9

# apt-get remove --purge --autoremove --yes ${DEV_PACKAGES}

RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cuda-dev nvidia-cudnn8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cudnn8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-tensorrt-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     nvidia-vpi-dev \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-opencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    zip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    hdf5-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#RUN pyenv global 3.9
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-dev
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir testresources setuptools==65.5.0
RUN pip install --no-cache-dir numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.7.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    liblapack-dev \
    libblas-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    libjpeg8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get download nvidia-l4t-jetson-multimedia-api \
    && dpkg-deb -R ./nvidia-l4t-jetson-multimedia-api_*_arm64.deb ./mm-api \
    && cp -r ./mm-api/usr/src/jetson_multimedia_api /usr/src/jetson_multimedia_api \
    && ./mm-api/DEBIAN/postinst \
    && rm -rf ./nvidia-l4t-jetson-multimedia-api_*_arm64.deb ./mm-api

RUN ldconfig

RUN apt-get install python-is-python3

RUN pip install --no-cache-dir 'absl-py>=1.0.0'
RUN pip install --no-cache-dir 'astunparse>=1.6.0'
RUN pip install --no-cache-dir 'flatbuffers>=23.5.26'
RUN pip install --no-cache-dir 'gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1'
RUN pip install --no-cache-dir 'google-pasta>=0.1.1'
RUN pip install --no-cache-dir 'h5py>=2.9.0'
RUN pip install --no-cache-dir 'libclang>=13.0.0'
RUN pip install --no-cache-dir 'ml-dtypes~=0.2.0'
RUN pip install --no-cache-dir 'numpy<2.0.0'
RUN pip install --no-cache-dir 'numpy<2.0.0,>=1.23.5'
RUN pip install --no-cache-dir 'opt-einsum>=2.3.2'
RUN pip install --no-cache-dir 'packaging'
RUN pip install --no-cache-dir 'protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3'
RUN pip install --no-cache-dir 'setuptools'
RUN pip install --no-cache-dir 'six>=1.12.0'
RUN pip install --no-cache-dir 'termcolor>=1.1.0'
RUN pip install --no-cache-dir 'typing-extensions>=3.6.6'
RUN pip install --no-cache-dir 'wrapt<1.15,>=1.11.0'
RUN pip install --no-cache-dir 'tensorflow-io-gcs-filesystem>=0.23.1'
RUN pip install --no-cache-dir 'grpcio<2.0,>=1.24.3'
RUN pip install --no-cache-dir 'tensorboard<2.16,>=2.15'
RUN pip install --no-cache-dir 'tensorflow-estimator<2.16,>=2.15.0'
RUN pip install --no-cache-dir 'keras<2.16,>=2.15.0'
RUN pip install --no-cache-dir nvidia-pyindex
#RUN pip install --no-cache-dir 'nvidia-cublas-cu12==12.2.5.6'
#RUN pip install --no-cache-dir 'nvidia-cuda-cupti-cu12==12.2.142'
#RUN pip install --no-cache-dir 'nvidia-cuda-nvcc-cu12==12.2.140'
#RUN pip install --no-cache-dir 'nvidia-cuda-nvrtc-cu12==12.2.140'
#RUN pip install --no-cache-dir 'nvidia-cuda-runtime-cu12==12.2.140'
#RUN pip install --no-cache-dir 'nvidia-cudnn-cu12==8.9.4.25'
#RUN pip install --no-cache-dir 'nvidia-cufft-cu12==11.0.8.103'
#RUN pip install --no-cache-dir 'nvidia-curand-cu12==10.3.3.141'
#RUN pip install --no-cache-dir 'nvidia-cusolver-cu12==11.5.2.141'
#RUN pip install --no-cache-dir 'nvidia-cusparse-cu12==12.1.2.141'
#RUN pip install --no-cache-dir 'nvidia-nccl-cu12==2.16.5'
#RUN pip install --no-cache-dir 'nvidia-nvjitlink-cu12==12.2.140'
#RUN pip install --no-cache-dir 'tensorrt==8.6.1.post1'
#RUN pip install --no-cache-dir 'tensorrt-bindings==8.6.1'
#RUN pip install --no-cache-dir 'tensorrt-libs==8.6.1'
RUN H5PY_SETUP_REQUIRES=0 pip install --no-cache-dir --verbose h5py
RUN pip install --no-cache-dir --verbose future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 futures pybind11

RUN pip install --no-cache-dir --index-url https://developer.download.nvidia.com/compute/redist/jp/v60dp/ tensorflow==2.15.0+nv24.02
#RUN pip install --no-cache-dir --index-url https://developer.download.nvidia.com/compute/redist/jp/v60dp/ pytorch==2.3.0a0+ebedce2.nv24.02
#RUN curl https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorflow-2.6.2+nv21.12-cp36-cp36m-linux_aarch64.whl > tensorflow-2.6.2+nv21.12-cp36-cp36m-linux_aarch64.whl
#RUN pip install --no-cache-dir tensorflow-2.6.2+nv21.12-cp36-cp36m-linux_aarch64.whl

#RUN curl https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl > torch.whl
#RUN pip install --no-cache-dir torch.whl

#RUN pip install --no-cache-dir setuptools Cython wheel

# Install Python TF dependencies

COPY scripts/* /usr/bin/

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN pip install --no-cache-dir gom

RUN apt-get update
RUN apt-get install --assume-yes --no-install-recommends \
    graphviz less

COPY scripts/* /usr/bin/

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:${LD_PRELOAD}

WORKDIR /data
