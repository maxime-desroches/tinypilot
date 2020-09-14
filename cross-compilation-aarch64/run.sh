#!/bin/bash

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

export PYTHONPATH="/openpilot"

cd /openpilot
scons -c
scons cc=1 new_compiler=1 -j$(nproc)
./cross-compilation-aarch64/install_package/create_install_package.sh

