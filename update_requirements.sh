#!/bin/bash -e
HOST="$(uname -m)"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [[ "$HOST" == "Darwin" ]]; then
  alias python="python3"
  alias pip="pip3"
fi

if ! command -v "pyenv" > /dev/null 2>&1; then
  echo "installing pyenv..."
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
  export PATH=$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH
fi

export MAKEFLAGS="-j$(nproc)"

PYENV_PYTHON_VERSION=$(cat .python-version)
if ! pyenv prefix ${PYENV_PYTHON_VERSION} &> /dev/null; then
  echo "pyenv ${PYENV_PYTHON_VERSION} install ..."
  CONFIGURE_OPTS="--enable-shared" pyenv install -f ${PYENV_PYTHON_VERSION}
fi

if ! command -v pipenv &> /dev/null; then
  echo "pipenv install ..."
  pip install pipenv
fi

echo "update pip"
pip install pip==21.3.1
pip install pipenv==2021.11.23

if [ -d "./xx" ]; then
  export PIPENV_SYSTEM=1
  export PIPENV_PIPFILE=./xx/Pipfile
fi

if [ -z "$PIPENV_SYSTEM" ]; then
  RUN="pipenv run"
else
  RUN=""
fi

echo "pip packages install for $HOST ..."
if [[ "$HOST" == "Darwin" ]]; then
  pipenv install --dev --deploy --clear --skip-lock
else
  pipenv install --dev --deploy --clear
fi
pyenv rehash

if [ -f "$DIR/.pre-commit-config.yaml" ]; then
  echo "precommit install ..."
  $RUN pre-commit install
  [ -d "./xx" ] && (cd xx && $RUN pre-commit install)
  [ -d "./notebooks" ] && (cd notebooks && $RUN pre-commit install)
fi
