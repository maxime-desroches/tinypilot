#!/usr/bin/env bash

TARGET_USER=batman
source .devcontainer/.host/.env

# override display flag for mac hosts
if [[ $HOST_OS == darwin ]]; then
  echo "Setting up DISPLAY override for macOS..."
  cat <<EOF >> /home/$TARGET_USER/.bashrc
if [ -n "\$DISPLAY" ]; then
  DISPLAY_NUM=\$(echo "\$DISPLAY" | awk -F: '{print \$NF}')
  export DISPLAY=host.docker.internal:\$DISPLAY_NUM
fi
EOF
fi

# setup virtualgl for mac hosts
if [[ $HOST_OS == darwin ]]; then
  echo "Setting up virtualgl for macOS..."
  cat <<EOF >> /home/$TARGET_USER/.bashrc
export VGL_PORT=10000
export VGL_CLIENT=host.docker.internal
export VGL_COMPRESS=rgb
export VGL_DISPLAY=:0
export VGL_FPS=60
EOF
fi

# These lines are temporary, to remain backwards compatible with old devcontainers
# that were running as root and therefore had their caches written as root
sudo chown -R $TARGET_USER: /tmp/scons_cache
sudo chown -R $TARGET_USER: /tmp/comma_download_cache
sudo chown -R $TARGET_USER: /home/batman/.comma
