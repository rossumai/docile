FROM python:3.10

RUN apt-get update

# Install opencv-python dependency
RUN apt-get install libgl1 -y

# Install poppler for pdf2image (converting pdf to images)
RUN apt-get install poppler-utils -y

# Install poetry
RUN pip3 install poetry

WORKDIR /app
COPY poetry.lock pyproject.toml /app/

COPY docile /app/docile
COPY README.md /app/README.md

RUN poetry install --no-interaction --with test --with doctr
