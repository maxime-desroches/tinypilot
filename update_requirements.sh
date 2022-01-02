#!/bin/bash -e

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null

if ! command -v "pyenv" > /dev/null 2>&1; then
  echo "installing pyenv..."
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
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
pip install pipenv==2021.5.29

echo "pip packages install ..."
if [ -d "./xx" ]; then
  export PIPENV_PIPFILE=./xx/Pipfile
  pipenv install --system --dev --deploy
  RUN=""
else
  pipenv install --dev --deploy
  RUN="pipenv run"
fi

echo "precommit install ..."
$RUN pre-commit install

# for internal comma repos
[ -d "./xx" ] && (cd xx && $RUN pre-commit install)
[ -d "./notebooks" ] && (cd notebooks && $RUN pre-commit install)

# update shims for newly installed executables (e.g. scons)
pyenv rehash
