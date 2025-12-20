#!/bin/bash
# Apple Silicon setup with pyenv
set -e

# ensure pyenv + pyenv-virtualenv are installed
if ! command -v pyenv &> /dev/null; then
  echo "pyenv not found. Please install pyenv first."; exit 1
fi

PYTHON_VERSION=3.10.14
ENV_NAME=itrlm-rag-310

pyenv install -s $PYTHON_VERSION
pyenv virtualenv $PYTHON_VERSION $ENV_NAME || true
pyenv activate $ENV_NAME

python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "✅ Environment $ENV_NAME ready with Python $PYTHON_VERSION"
