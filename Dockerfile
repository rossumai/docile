FROM python:3.10

# cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# install screen
RUN apt-get install screen -y
RUN echo "defshell -bash" > ~/.screenrc

# Install poetry
RUN pip3 install poetry

WORKDIR /app
COPY poetry.lock pyproject.toml /app/

RUN poetry install --no-interaction --with test --with doctr
