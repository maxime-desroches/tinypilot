#!/usr/bin/bash -e

BUILD_DIR=/tmp/releasepilot
SOURCE_DIR=/data/openpilot

ln -snf $BUILD_DIR /data/pythonpath

export GIT_COMMITTER_NAME="Vehicle Researcher"
export GIT_COMMITTER_EMAIL="user@comma.ai"
export GIT_AUTHOR_NAME="Vehicle Researcher"
export GIT_AUTHOR_EMAIL="user@comma.ai"
export GIT_SSH_COMMAND="ssh -i /data/gitkey"

echo "[-] Setting up repo T=$SECONDS"
#rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR
git init
git remote add origin git@github.com:commaai/openpilot.git || true

echo "[-] fetching public T=$SECONDS"
git prune || true
git remote prune origin || true

echo "[-] bringing master-ci and devel in sync T=$SECONDS"
git fetch origin master-ci
git fetch origin devel

git checkout -f -B master-ci
git reset --hard origin/devel
git clean -xdf

# remove everything except .git
echo "[-] erasing old files T=$SECONDS"
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;

# reset tree and get version
cd $SOURCE_DIR
#git clean -xdf
#git checkout -- selfdrive/common/version.h

VERSION=$(cat selfdrive/common/version.h | awk -F\" '{print $2}')
echo "#define COMMA_VERSION \"$VERSION-$(git --git-dir=$SOURCE_DIR/.git rev-parse --short HEAD)-$(date '+%Y-%m-%dT%H:%M:%S')\"" > selfdrive/common/version.h

# do the files copy
echo "[-] copying files T=$SECONDS"
cd $SOURCE_DIR
cp -pR --parents $(cat release/files_common) $BUILD_DIR/

# in the directory
cd $BUILD_DIR

rm -f panda/board/obj/panda.bin.signed

echo "[-] committing version $VERSION T=$SECONDS"
git add -f .
git status
git commit -a -m "openpilot v$VERSION release"

# Build signed panda firmware
pushd panda/board/
cp -r /data/pandaextra /data/openpilot/
RELEASE=1 make obj/panda.bin
mv obj/panda.bin /tmp/panda.bin
make clean
mv /tmp/panda.bin obj/panda.bin.signed
rm -rf /data/openpilot/pandaextra
popd

# Build
export PYTHONPATH="$BUILD_DIR"
SCONS_CACHE=1 scons -j$(nproc)

# Run tests
#python selfdrive/test/test_manager.py
selfdrive/car/tests/test_car_interfaces.py

# Cleanup
find . -name '*.a' -delete
find . -name '*.o' -delete
find . -name '*.os' -delete
find . -name '*.pyc' -delete
find . -name '__pycache__' -delete
rm -rf .sconsign.dblite Jenkinsfile release/

# Restore phonelibs
git checkout phonelibs/

# Mark as prebuilt release
touch prebuilt

# Add built files to git
git add -f .
git commit --amend -m "openpilot v$VERSION"

if [ ! -z "$PUSH" ]; then
  git remote set-url origin git@github.com:commaai/openpilot.git

  # Push to release2-staging
  git push origin release3-staging

  # Create dashcam release
  git rm selfdrive/car/*/carcontroller.py

  git commit -m "create dashcam release from release"
  git push origin release3-staging:dashcam3-staging
fi

echo "[-] done T=$SECONDS"
