FROM python:3.10

RUN apt-get update

# Install poppler for pdf2image (converting pdf to images)
RUN apt-get install poppler-utils -y

# Install poetry
RUN pip3 install poetry

WORKDIR /app
COPY poetry.lock pyproject.toml /app/

COPY docile /app/docile
COPY LICENSE /app/LICENSE
COPY README.md /app/README.md

RUN poetry install --no-interaction --extras interactive
