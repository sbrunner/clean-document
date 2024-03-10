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

RUN pyenv global 3.9
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir testresources setuptools==65.5.0
RUN pip install --no-cache-dir numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.7.0
#RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 tensorflow==2.12.0+nv23.06
RUN python3 --version && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.6.2+nv21.12

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

RUN pip3 install --no-cache-dir setuptools Cython wheel

# install Python TF dependencies
RUN H5PY_SETUP_REQUIRES=0 pip install --no-cache-dir --verbose h5py
RUN pip3 install --no-cache-dir --verbose future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 futures pybind11

# TensorFlow
ARG TENSORFLOW_URL
ARG TENSORFLOW_WHL

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${TENSORFLOW_URL} -O ${TENSORFLOW_WHL} \
    && pip3 install --no-cache-dir --verbose ${TENSORFLOW_WHL} \
    && rm ${TENSORFLOW_WHL}

COPY scripts/* /usr/bin/

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN pip install nvidia-pyindex
RUN pip install --upgrade nvidia-tensorrt
RUN pip install torch gom

RUN apt-get update
RUN apt-get install --assume-yes --no-install-recommends \
    graphviz less

COPY scripts/* /usr/bin/

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:${LD_PRELOAD}

WORKDIR /data
