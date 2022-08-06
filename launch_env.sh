#!/usr/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="5.2"
fi

if [ -z "$PASSIVE" ]; then
  export PASSIVE="1"
fi

# FIXME -- remove this
export FINGERPRINT="HONGQI HS5 1ST GEN"

export STAGING_ROOT="/data/safe_staging"
