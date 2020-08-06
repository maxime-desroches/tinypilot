#!/usr/bin/env python3

# Safe Update: A simple service that waits for network access and tries to
# update every 10 minutes. It's intended to make the OP update process more
# robust against Git repository corruption. This service DOES NOT try to fix
# an already-corrupt BASEDIR Git repo, only prevent it from happening.
#
# During normal operation, both onroad and offroad, the update process makes
# no changes to the BASEDIR install of OP. All update attempts are performed
# in a disposable staging area provided by OverlayFS. It assumes the deleter
# process provides enough disk space to carry out the process.
#
# If an update succeeds, a flag is set, and the update is swapped in at the
# next reboot. If an update is interrupted or otherwise fails, the OverlayFS
# upper layer and metadata can be discarded before trying again.
#
# The swap on boot is triggered by launch_chffrplus.sh
# gated on the existence of $FINALIZED/.overlay_consistent and also the
# existence and mtime of $BASEDIR/.overlay_init.
#
# Other than build byproducts, BASEDIR should not be modified while this
# service is running. Developers modifying code directly in BASEDIR should
# disable this service.

import os
import datetime
import subprocess
import psutil
import shutil
import signal
import fcntl
import time
import threading
from cffi import FFI
from pathlib import Path

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.alertmanager import set_offroad_alert

STAGING_ROOT = "/data/safe_staging"

OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")
FINALIZED = os.path.join(STAGING_ROOT, "finalized")

NEOSUPDATE_DIR = "/data/neosupdate"


# Workaround for lack of os.link in the NEOS/termux python
ffi = FFI()
ffi.cdef("int link(const char *oldpath, const char *newpath);")
libc = ffi.dlopen(None)
def link(src, dest):
  return libc.link(src.encode(), dest.encode())


class WaitTimeHelper:
  def __init__(self):
    self.ready_event = threading.Event()
    self.shutdown = False
    signal.signal(signal.SIGTERM, self.graceful_shutdown)
    signal.signal(signal.SIGINT, self.graceful_shutdown)
    signal.signal(signal.SIGHUP, self.update_now)

  def graceful_shutdown(self, signum, frame):
    # umount -f doesn't appear effective in avoiding "device busy" on NEOS,
    # so don't actually die until the next convenient opportunity in main().
    cloudlog.info("caught SIGINT/SIGTERM, dismounting overlay at next opportunity")
    self.shutdown = True
    self.ready_event.set()

  def update_now(self, signum, frame):
    cloudlog.info("caught SIGHUP, running update check immediately")
    self.ready_event.set()

  def sleep(self, t):
    self.ready_event.wait(timeout=t)


def run(cmd, cwd=None, low_priority=False):
  if low_priority:
    cmd = ["nice", "-n", "19"] + cmd
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')


def remove_consistent_flag():
  os.system("sync")
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  try:
    consistent_file.unlink()
  except FileNotFoundError:
    pass
  os.system("sync")


def set_consistent_flag():
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  os.system("sync")
  consistent_file.touch()
  os.system("sync")


def set_update_available_params(new_version):
  params = Params()

  t = datetime.datetime.utcnow().isoformat()
  params.put("LastUpdateTime", t.encode('utf8'))

  if new_version:
    try:
      with open(os.path.join(FINALIZED, "RELEASES.md"), "rb") as f:
        r = f.read()
      r = r[:r.find(b'\n\n')]  # Slice latest release notes
      params.put("ReleaseNotes", r + b"\n")
    except Exception:
      params.put("ReleaseNotes", "")
    params.put("UpdateAvailable", "1")


def dismount_ovfs():
  if os.path.ismount(OVERLAY_MERGED):
    cloudlog.error("unmounting existing overlay")
    run(["umount", "-l", OVERLAY_MERGED])


def setup_git_options(cwd):
  # We sync FS object atimes (which NEOS doesn't use) and mtimes, but ctimes
  # are outside user control. Make sure Git is set up to ignore system ctimes,
  # because they change when we make hard links during finalize. Otherwise,
  # there is a lot of unnecessary churn. This appears to be a common need on
  # OSX as well: https://www.git-tower.com/blog/make-git-rebase-safe-on-osx/

  # We are using copytree to copy the directory, which also changes
  # inode numbers. Ignore those changes too.
  git_cfg = [
    ("core.trustctime", "false"),
    ("core.checkStat", "minimal"),
  ]
  for option, value in git_cfg:
    try:
      ret = run(["git", "config", "--get", option], cwd)
      config_ok = (ret.strip() == value)
    except subprocess.CalledProcessError:
      config_ok = False

    if not config_ok:
      cloudlog.info(f"Setting git '{option}' to '{value}'")
      run(["git", "config", option, value], cwd)


def init_ovfs():
  cloudlog.info("preparing new safe staging area")
  Params().put("UpdateAvailable", "0")

  remove_consistent_flag()

  dismount_ovfs()
  if os.path.isdir(STAGING_ROOT):
    shutil.rmtree(STAGING_ROOT)

  for dirname in [STAGING_ROOT, OVERLAY_UPPER, OVERLAY_METADATA, OVERLAY_MERGED, FINALIZED]:
    os.mkdir(dirname, 0o755)
  if not os.lstat(BASEDIR).st_dev == os.lstat(OVERLAY_MERGED).st_dev:
    raise RuntimeError("base and overlay merge directories are on different filesystems; not valid for overlay FS!")

  # Remove consistent flag from current BASEDIR so it's not copied over
  if os.path.isfile(os.path.join(BASEDIR, ".overlay_consistent")):
    os.remove(os.path.join(BASEDIR, ".overlay_consistent"))

  # Leave a timestamped canary in BASEDIR to check at startup. The device clock
  # should be correct by the time we get here. If the init file disappears, or
  # critical mtimes in BASEDIR are newer than .overlay_init, continue.sh can
  # assume that BASEDIR has used for local development or otherwise modified,
  # and skips the update activation attempt.
  Path(os.path.join(BASEDIR, ".overlay_init")).touch()

  overlay_opts = f"lowerdir={BASEDIR},upperdir={OVERLAY_UPPER},workdir={OVERLAY_METADATA}"
  run(["mount", "-t", "overlay", "-o", overlay_opts, "none", OVERLAY_MERGED])


def finalize_from_ovfs():
  """Take the current OverlayFS merged view and finalize a copy outside of
  OverlayFS, ready to be swapped-in at BASEDIR. Copy using shutil.copytree"""

  cloudlog.info("creating finalized version of the overlay")
  shutil.rmtree(FINALIZED)
  shutil.copytree(OVERLAY_MERGED, FINALIZED, symlinks=True)
  cloudlog.info("done finalizing overlay")


def attempt_update(wait_helper):
  cloudlog.info("attempting git update inside staging overlay")

  setup_git_options(OVERLAY_MERGED)

  git_fetch_output = run(["git", "fetch"], OVERLAY_MERGED, low_priority=True)
  cloudlog.info("git fetch success: %s", git_fetch_output)

  cur_hash = run(["git", "rev-parse", "HEAD"], OVERLAY_MERGED).rstrip()
  upstream_hash = run(["git", "rev-parse", "@{u}"], OVERLAY_MERGED).rstrip()
  new_version = cur_hash != upstream_hash

  err_msg = "Failed to add the host to the list of known hosts (/data/data/com.termux/files/home/.ssh/known_hosts).\n"
  git_fetch_result = len(git_fetch_output) > 0 and (git_fetch_output != err_msg)

  cloudlog.info("comparing %s to %s" % (cur_hash, upstream_hash))
  if new_version or git_fetch_result:
    cloudlog.info("Running update")
    if new_version:
      cloudlog.info("git reset in progress")
      r = [
        run(["git", "reset", "--hard", "@{u}"], OVERLAY_MERGED, low_priority=True),
        run(["git", "clean", "-xdf"], OVERLAY_MERGED, low_priority=True ),
        run(["git", "submodule", "init"], OVERLAY_MERGED, low_priority=True),
        run(["git", "submodule", "update"], OVERLAY_MERGED, low_priority=True),
      ]
      cloudlog.info("git reset success: %s", '\n'.join(r))

      # Download the accompanying NEOS version if it doesn't match the current version
      with open("/VERSION", "r") as f:
        current_neos_version = f.read().strip()

      required_neos_version = run(["bash", "-c",
                                   r"unset REQUIRED_NEOS_VERSION && source launch_env.sh && echo -n $REQUIRED_NEOS_VERSION"],
                                   OVERLAY_MERGED).strip()

      cloudlog.info(f"NEOS version update check: {current_neos_version} current, {required_neos_version} in update")
      if current_neos_version != required_neos_version:
        cloudlog.info(f"Beginning background download for NEOS {required_neos_version}")

        update_manifest = f'file:///{OVERLAY_MERGED}/installer/updater/update.json'
        set_offroad_alert("Offroad_NeosUpdate", True)

        neos_downloaded = False
        start_time = time.monotonic()
        # Try to download for one day
        while (time.monotonic() - start_time < 60*60*24) and not wait_helper.shutdown:
          try:
            updater_path = os.path.join(OVERLAY_MERGED, "installer/updater/updater")
            run([updater_path, "bgcache", update_manifest], OVERLAY_MERGED, low_priority=True)

            cloudlog.info("NEOS background download successful, took {time.monotonic() - start_time} seconds")
            neos_downloaded = True
            break
          except subprocess.CalledProcessError:
            cloudlog.info("NEOS background download failed, retrying")
            if wait_helper.sleep(120):
              break

        # If the download failed, we'll show the alert again when we retry
        set_offroad_alert("Offroad_NeosUpdate", False)
        if not neos_downloaded:
          raise Exception("Failed to download NEOS update")

    # Un-set the validity flag to prevent the finalized tree from being
    # activated later if the finalize step is interrupted
    remove_consistent_flag()

    finalize_from_ovfs()

    # Make sure the validity flag lands on disk LAST, only when the local git
    # repo and OP install are in a consistent state.
    set_consistent_flag()

    cloudlog.info("openpilot update successful!")
  else:
    cloudlog.info("nothing new from git at this time")

  # Clear old NEOS updates
  #if not new_version and os.path.isdir(NEOSUPDATE_DIR):
  #  shutil.rmtree(NEOSUPDATE_DIR)

  set_update_available_params(new_version)
  return new_version


def main():
  update_failed_count = 0
  overlay_init_done = False
  params = Params()

  if params.get("DisableUpdates") == b"1":
    raise RuntimeError("updates are disabled by the DisableUpdates param")

  if os.geteuid() != 0:
    raise RuntimeError("updated must be launched as root!")

  # Set low io priority
  p = psutil.Process()
  if psutil.LINUX:
    p.ionice(psutil.IOPRIO_CLASS_BE, value=7)

  ov_lock_fd = open('/tmp/safe_staging_overlay.lock', 'w')
  try:
    fcntl.flock(ov_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except IOError:
    raise RuntimeError("couldn't get overlay lock; is another updated running?")

  # Wait for IsOffroad to be set before our first update attempt
  wait_helper = WaitTimeHelper()
  wait_helper.sleep(30)

  while not wait_helper.shutdown:
    update_failed_count += 1
    wait_helper.ready_event.clear()

    # Check for internet every 30s
    time_wrong = datetime.datetime.utcnow().year < 2019
    ping_failed = subprocess.call(["ping", "-W", "4", "-c", "1", "8.8.8.8"])
    if ping_failed or time_wrong:
      wait_helper.sleep(30)
      continue

    # Attempt an update
    try:
      # If the git directory has modifcations after we created the overlay
      # we need to recreate the overlay
      if overlay_init_done:
        overlay_init_fn = os.path.join(BASEDIR, ".overlay_init")
        git_dir_path = os.path.join(BASEDIR, ".git")
        new_files = run(["find", git_dir_path, "-newer", overlay_init_fn])

        if len(new_files.splitlines()):
          cloudlog.info(".git directory changed, recreating overlay")
          overlay_init_done = False

      if not overlay_init_done:
        init_ovfs()
        overlay_init_done = True

      if params.get("IsOffroad") == b"1":
        attempt_update(wait_helper)
        update_failed_count = 0
      else:
        cloudlog.info("not running updater, openpilot running")

    except subprocess.CalledProcessError as e:
      cloudlog.event(
        "update process failed",
        cmd=e.cmd,
        output=e.output,
        returncode=e.returncode
      )
      overlay_init_done = False
    except Exception:
      cloudlog.exception("uncaught updated exception, shouldn't happen")

    params.put("UpdateFailedCount", str(update_failed_count))
    wait_helper.sleep(60*10)

  dismount_ovfs()

if __name__ == "__main__":
  main()
