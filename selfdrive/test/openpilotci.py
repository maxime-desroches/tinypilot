#!/usr/bin/env python3
import os
import sys
import subprocess

ACCOUNT_URL = "https://commadataci.blob.core.windows.net"
CONTAINER_NAME = "openpilotci"
BASE_URL = f"{ACCOUNT_URL}/{CONTAINER_NAME}/"
TOKEN_PATH = "/data/azure_token"


def get_url(route_name, segment_num, log_type="rlog"):
  ext = "hevc" if log_type.endswith('camera') else "bz2"
  return BASE_URL + f"{route_name.replace('|', '/')}/{segment_num}/{log_type}.{ext}"

def get_sas_token():
  sas_token = os.environ.get("AZURE_TOKEN", None)
  if os.path.isfile(TOKEN_PATH):
    sas_token = open(TOKEN_PATH).read().strip()

  if sas_token is None:
    sas_token = subprocess.check_output("az storage container generate-sas --account-name commadataci --name openpilotci \
                                         --https-only --permissions lrw --expiry $(date -u '+%Y-%m-%dT%H:%M:%SZ' -d '+1 hour') \
                                         --auth-mode login --as-user --output tsv", shell=True).decode().strip("\n")

  return sas_token

def upload_bytes(data, name):
  from azure.storage.blob import BlobServiceClient
  service = BlobServiceClient(ACCOUNT_URL, credential=get_sas_token())
  blob = service.get_blob_client(container=CONTAINER_NAME, blob=name)
  blob.upload_blob(data)
  return BASE_URL + name

def upload_file(path, name):
  with open(path, "rb") as f:
    return upload_bytes(f.read(), name)


if __name__ == "__main__":
  for f in sys.argv[1:]:
    name = os.path.basename(f)
    url = upload_file(f, name)
    print(url)
