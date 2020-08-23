#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../../

docker pull docker.io/commaai/openpilot-sim:latest || true
docker build --cache-from docker.io/commaai/openpilot-sim:latest -t commaai/openpilot-sim:latest -f tools/sim/Dockerfile.sim .
