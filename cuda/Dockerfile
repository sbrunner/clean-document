# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base
# https://repo.download.nvidia.com/jetson/
# Jetpack 4.6.4
# L4T 32.7.4
# jetson_release

FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3 as base

SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

#RUN python3 -c 'import cv2; print(cv2.__version__)'
#RUN python3 -c 'import tensorflow; print(tensorflow.__version__)'

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get install --assume-yes --no-install-recommends \
    graphviz less python3-tk python3-dev cmake

RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install scikit-build
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install keras==2.3.1 opencv-python-headless

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get remove --purge --autoremove --yes \
    build-essential python3-dev cmake

RUN apt-get clean

RUN rm -rf /root/.cache /root/.cargo /root/.jupyter /root/.keras /root/.rustup /root/.wget-hsts

#RUN python3 -c 'import cv2; print(cv2.__version__)'
#RUN python3 -c 'import tensorflow; print(tensorflow.__version__)'

COPY scripts/* /usr/bin/

FROM scratch as clean_document
COPY --from=base / /

WORKDIR /data
