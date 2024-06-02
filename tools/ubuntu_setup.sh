#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

export SUDO=""

# Use sudo if not root
if [[ ! $(id -u) -eq 0 ]]; then
  if [[ -z $(which sudo) ]]; then
    echo "Please install sudo or run as root"
    exit 1
  fi
  SUDO="sudo"
fi

clear
echo " --   WELCOME TO THE OPENPILOT SETUP   --"
echo
echo "-- sudo is required for apt installation --"


# NOTE: this is used in a docker build, so do not run any scripts here.

$DIR/install_ubuntu_dependencies.sh
$DIR/install_python_dependencies.sh

echo
echo "----   OPENPILOT SETUP DONE   ----"
echo "Open a new shell or configure your active shell env by running:"
echo "source ~/.bashrc"
echo
echo "To activate your virtual env using poetry, run either:"
echo
echo "`poetry shell` or `.venv/bin/activate`"
