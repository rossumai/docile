# FROM python:3.10
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install python3.10 python3-pip -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# Install poetry
RUN python3 -m pip install poetry

# cv2 dependencies
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Install opencv-python dependency
# RUN apt-get install libgl1 -y

# Install poppler for pdf2image (converting pdf to images)
RUN apt-get install poppler-utils -y

# # Install poetry
# RUN pip3 install poetry

WORKDIR /app
COPY poetry.lock pyproject.toml /app/

COPY docile /app/docile
COPY baselines /app/baselines
COPY README.md /app/README.md

# RUN poetry install --no-interaction --with test --with doctr
RUN poetry install --no-interaction --with test --with doctr --with baselines
