#!/usr/bin/bash

set -e

if [ -z "$SOURCE_DIR" ]; then
  echo "SOURCE_DIR must be set"
  exit 1
fi

if [ -z "$GIT_COMMIT" ]; then
  echo "GIT_COMMIT must be set"
  exit 1
fi

if [ -z "$TEST_DIR" ]; then
  echo "TEST_DIR must be set"
  exit 1
fi

umount /data/safe_staging/merged/ || true
sudo umount /data/safe_staging/merged/ || true

if [ -f "/EON" ]; then
  rm -rf /data/core
  rm -rf /data/neoupdate
  rm -rf /data/safe_staging
fi

export KEYS_PARAM_PATH="/usr/comma/setup_keys"
if [ -f "/EON" ]; then
  export KEYS_PATH="/data/data/com.termux/files/home/setup_keys"
  export CONTINUE_PATH="/data/data/com.termux/files/continue.sh"

  if ! grep -F "$KEYS_PATH" /usr/etc/ssh/sshd_config; then
    echo "setting up keys"
    mount -o rw,remount /system
    sed -i 's,$KEYS_PARAM_PATH,$KEYS_PATH,' /usr/etc/ssh/sshd_config
    mount -o ro,remount /system
  fi
else
  export KEYS_PATH="/usr/comma/setup_keys"
  export CONTINUE_PATH="/data/continue.sh"

  if ! grep -F "$KEYS_PATH" /etc/ssh/sshd_config; then
    echo "setting up keys"
    mount -o rw,remount /
    sed -i 's,$KEYS_PARAM_PATH,$KEYS_PATH,' /etc/ssh/sshd_config
    mount -o ro,remount /
  fi
fi

tee $CONTINUE_PATH << EOF
#!/usr/bin/bash

PARAMS_ROOT="/data/params/d"

while true; do
  mkdir -p \$PARAMS_ROOT
  echo -n 1 > \$PARAMS_ROOT/SshEnabled
  sleep 1m
done

sleep infinity
EOF
chmod +x $CONTINUE_PATH

# set up environment
if [ ! -d "$SOURCE_DIR" ]; then
  git clone https://github.com/commaai/openpilot.git $SOURCE_DIR
fi
cd $SOURCE_DIR

rm -f .git/index.lock
git reset --hard
git fetch
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;
git fetch --verbose origin $GIT_COMMIT
git reset --hard $GIT_COMMIT
git checkout $GIT_COMMIT
git clean -xdf
git submodule update --init --recursive
git submodule foreach --recursive "git reset --hard && git clean -xdf"

echo "git checkout done, t=$SECONDS"

rsync -a --delete $SOURCE_DIR $TEST_DIR

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
