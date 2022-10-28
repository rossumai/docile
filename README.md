# DocILE
DocILE: Document Information Localization and Extraction Benchmark

# Installation
You need to install some dependencies for PDF handling:

- Ubuntu
```bash
apt-get install poppler-utils -y
```
- Mac OS X:
```bash
brew install poppler
```

To install the DocTR with, follow the tutorial at:

- Mac OS X:
```bash
brew install cairo pango gdk-pixbuf libffi

pip install "python-doctr[tf]"
```

# Development instructions
We use https://python-poetry.org/docs/ for dependency management. Follow the official instructions for to install Poetry. To install the dependencies and activate the virtual environment use:

```bash
poetry install
```

Note that you can use `poetry shell` to spawn a new shell with this virtual environment activated.

You will also need https://pre-commit.com/ for pre-commit checks. If you have done the steps above, it is already installed in your virtual environment. To activate the pre-commit, use:

```bash
pre-commit install
```
