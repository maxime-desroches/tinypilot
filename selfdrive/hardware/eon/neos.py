#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
import os
import requests

NEOSUPDATE_DIR = "/data/neoupdate"

RECOVERY_DEV = "/dev/block/bootdevice/by-name/recovery"
RECOVERY_COMMAND = "/cache/recovery/command"


def download_file(url: str, fn: str, sha256: str, display_name: str, cloudlog) -> None:
  # check if already downloaded
  if check_hash(fn, sha256):
    cloudlog.info(f"{display_name} already cached")
    return

  with open(fn, "ab+") as f:
    headers = {"Range": f"bytes={f.tell()}-"}
    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)

    total = int(r.headers['Content-Length'])
    if 'Content-Range' in r.headers:
      total = int(r.headers['Content-Range'].split('/')[-1])

    for chunk in r.iter_content(chunk_size=8192):
      f.write(chunk)
      print(f"Downloading {display_name}: {f.tell() / total * 100}")

  if not check_hash(fn, sha256):
    os.unlink(fn)
    raise Exception("downloaded update failed hash check")


def check_hash(fn: str, sha256: str, length: int = -1) -> bool:
  if not os.path.exists(fn):
    return False

  h = hashlib.sha256()
  with open(fn, "rb") as f:
    while True:
      print("reading", min(max(0, length - f.tell()), 8192), fn)
      dat = f.read(min(max(0, length - f.tell()), 8192))
      if not dat or f.tell() == length:
        break
      h.update(dat)
  return h.hexdigest() == sha256


def flash_update(update_fn: str, out_path: str) -> None:
  with open(update_fn, "rb") as update, open(out_path, "w+b") as out:
    while True:
      dat = update.read(8192)
      if len(dat) == 0:
        break
      out.write(dat)


def download_neos_update(manifest_path: str, cloudlog) -> None:
  with open(manifest_path) as f:
    m = json.load(f)

  os.makedirs(NEOSUPDATE_DIR, exist_ok=True)

  # handle recovery updates
  if not check_hash(RECOVERY_DEV, m['recovery_hash'], m['recovery_len']):
    cloudlog.info("recovery needs update")
    recovery_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['recovery_url']))
    download_file(m['recovery_url'], recovery_fn, m['recovery_hash'], "recovery", cloudlog)

    flash_update(recovery_fn, RECOVERY_DEV)
    assert check_hash(RECOVERY_DEV, m['recovery_hash'], m['recovery_len']), "recovery flash corrupted"
    cloudlog.info("recovery successfully flashed")

  # download OTA update
  ota_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['ota_url']))
  download_file(m['ota_url'], ota_fn, m['ota_hash'], "system", cloudlog)


def verify_update_ready(manifest_path: str) -> bool:
  with open(manifest_path) as f:
    m = json.load(f)

  ota_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['ota_url']))
  ota_downloaded = check_hash(ota_fn, m['ota_hash'])
  recovery_flashed = check_hash(RECOVERY_DEV, m['recovery_hash'], m['recovery_len'])
  return ota_downloaded and recovery_flashed


def perform_ota_update(manifest_path: str) -> None:
  with open(manifest_path) as f:
    m = json.load(f)

  # reboot into recovery
  ota_fn = os.path.join(NEOSUPDATE_DIR, os.path.basename(m['ota_url']))
  with open(RECOVERY_COMMAND, "w") as rf:
    rf.write(f"--update_package={ota_fn}\n")
  os.system("service call power 16 i32 0 s16 recovery i32 1")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="NEOS update utility",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--swap", action="store_true", help="Peform update after downloading")
  parser.add_argument("--swap-if-ready", action="store_true", help="Perform update if already downloaded")
  parser.add_argument("manifest", help="Manifest json")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)

  if args.swap_if_ready:
    if verify_update_ready(args.manifest):
      perform_ota_update(args.manifest)
  else:
    download_neos_update(args.manifest, logging)
    if args.swap:
      perform_ota_update(args.manifest)
