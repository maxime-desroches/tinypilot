#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(cd $DIR/../ && pwd)"

# Install brew if required
if [[ $(command -v brew) == "" ]]; then
  echo "-- Installing Hombrew"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
  echo "-- [✔] installed brew t=$SECONDS"
fi

# TODO: remove protobuf,protobuf-c,swig when casadi can be pip installed
echo "-- installing brew packages ..."
brew bundle --file=- <<-EOS
brew "cmake"
brew "cppcheck"
brew "git-lfs"
brew "zlib"
brew "bzip2"
brew "capnp"
brew "coreutils"
brew "eigen"
brew "ffmpeg"
brew "glfw"
brew "libarchive"
brew "libusb"
brew "libtool"
brew "llvm"
brew "curl"
brew "openssl"
brew "qt@5"
brew "zeromq"
brew "protobuf"
brew "protobuf-c"
brew "swig"
cask "gcc-arm-embedded"
EOS

echo "-- [✔] finished brew install t=$SECONDS"

if [[ $SHELL == "/bin/zsh" ]]; then
  RC_FILE="$HOME/.zshrc"
elif [[ $SHELL == "/bin/bash" ]]; then
  RC_FILE="$HOME/.bash_profile"
fi

BREW_PREFIX=$(brew --prefix)

# archive backend tools for pip dependencies
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/zlib/lib"
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/bzip2/lib"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/zlib/include"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/bzip2/include"

# pycurl curl/openssl backend dependencies
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/openssl@3/lib"
export LDFLAGS="$LDFLAGS -L${BREW_PREFIX}/opt/curl/lib"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/openssl@3/include"
export CPPFLAGS="$CPPFLAGS -I${BREW_PREFIX}/opt/curl/include"
export PATH="${BREW_PREFIX}/opt/openssl@3/bin:$PATH"
export PATH="${BREW_PREFIX}/opt/curl/bin:$PATH"

# newer (brew-installed) curl doesn't use LibreSSL backend
export CURL_SSL_BACKEND=secure-transport

# openpilot environment
if [ -z "$OPENPILOT_ENV" ] && [ -n "$RC_FILE" ] && [ -z "$CI" ]; then
  echo "export PATH=${BREW_PREFIX}/opt/openssl@3/bin:$"PATH"" >> $RC_FILE
  echo "export PATH=${BREW_PREFIX}/opt/curl/bin:$"PATH"" >> $RC_FILE
  echo "export CURL_SSL_BACKEND=secure-transport" >> $RC_FILE
  echo "source $ROOT/tools/openpilot_env.sh" >> $RC_FILE
  source "$ROOT/tools/openpilot_env.sh"
  echo "-- [✔] Added openpilot env params to RC file: $RC_FILE"
fi

# install python & dependencies
$ROOT/update_requirements.sh
eval "$(pyenv init --path)"
echo "-- [✔] installed python dependencies t=$SECONDS"

# install casadi
VENV=`pipenv --venv`
PYTHON_VER=3.8
PYTHON_VERSION=$(cat $ROOT/.python-version)
if [ ! -f "$VENV/include/casadi/casadi.hpp" ]; then
  echo "-- casadi manual install"
  cd /tmp/ && curl -L https://github.com/casadi/casadi/archive/refs/tags/ge6.tar.gz --output casadi.tar.gz
  tar -xzf casadi.tar.gz
  cd casadi-ge6/ && mkdir -p build && cd build
  cmake .. \
    -DWITH_PYTHON=ON \
    -DWITH_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=$VENV \
    -DPYTHON_PREFIX:PATH=$VENV/lib/python$PYTHON_VER/site-packages \
    -DPYTHON_LIBRARY:FILEPATH=$HOME/.pyenv/versions/$PYTHON_VERSION/lib/libpython$PYTHON_VER.dylib \
    -DPYTHON_EXECUTABLE:FILEPATH=$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python \
    -DPYTHON_INCLUDE_DIR:PATH=$HOME/.pyenv/versions/$PYTHON_VERSION/include/python$PYTHON_VER \
    -DCMAKE_CXX_FLAGS="-ferror-limit=0" -DCMAKE_C_FLAGS="-ferror-limit=0"
  CFLAGS="-ferror-limit=0" make -j$(nproc) && make install
  echo "-- [✔] installed casadi t=$SECONDS"
else
  echo "----   casadi found in venv. skipping build   ----"
fi

echo
echo "----   OPENPILOT SETUP DONE   ----"
echo "Open a new shell or configure your active shell env by running:"
echo "source $RC_FILE"
