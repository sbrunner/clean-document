FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3 as base

SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

# Install system deps
RUN apt-get update
RUN apt-get install --assume-yes --no-install-recommends \
    git graphviz less

# Install deps
WORKDIR /app
RUN python3 -m pip freeze > /requirements-org.txt
RUN python3 -m pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export --with=dev --without-hashes --output=requirements.txt
RUN python3 -m pip install --no-deps --requirement=requirements.txt
RUN python3 -m pip freeze > /requirements-dst.txt
RUN diff /requirements-org.txt /requirements-dst.txt > /requirements-diff.txt || true

# Install the app
WORKDIR /app
RUN python3 -m pip install --upgrade pip
COPY clean_document ./clean_document
COPY pyproject.toml poetry.lock README.md ./
RUN python3 -m pip install --no-deps --editable=.

COPY scripts/* /usr/bin/

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:${LD_PRELOAD}

WORKDIR /data
