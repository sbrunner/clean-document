# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base
# https://repo.download.nvidia.com/jetson/
# Jetpack 4.6.4
# L4T 32.7.4
# jetson_release

FROM nvcr.io/nvidia/tensorflow-pb23h2-igx:23.08.06-tf2-py3-dgpu as base

SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

#RUN python3 -c 'import cv2; print(cv2.__version__)'
#RUN python3 -c 'import tensorflow; print(tensorflow.__version__)'

RUN apt-get update
RUN apt-get install --assume-yes --no-install-recommends \
    graphviz less python3-tk

RUN apt-get remove --purge --autoremove --yes \
    build-essential

RUN apt-get clean

RUN python3 -m pip install keras==2.3.1

RUN python3 -m pip uninstall --yes torch torchvision

RUN rm -rf /root/.cache /root/.cargo /root/.jupyter /root/.keras /root/.rustup /root/.wget-hsts

#RUN python3 -c 'import cv2; print(cv2.__version__)'
#RUN python3 -c 'import tensorflow; print(tensorflow.__version__)'

COPY scripts/* /usr/bin/

FROM scratch as clean_document
COPY --from=base / /

WORKDIR /data
