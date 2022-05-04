#!/usr/bin/env python3
import os
import sys
import subprocess

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"
TOKEN_PATH = "/data/azure_token"


def get_url(route_name, segment_num, log_type="rlog"):
  ext = "hevc" if log_type.endswith('camera') else "bz2"
  return BASE_URL + f"{route_name.replace('|', '/')}/{segment_num}/{log_type}.{ext}"


def upload_file(path, name):
  from azure.storage.blob import BlockBlobService  # pylint: disable=import-error

  print('CI', os.getenv('CI'))
  print('FILEREADER_CACHE', os.getenv('FILEREADER_CACHE'))
  print('AZURE_TOKEN_IN_ENV', "AZURE_TOKEN" in os.environ)
  print('AZURE_TOKEN_TYPE', type(os.environ["AZURE_TOKEN"]))
  if "AZURE_TOKEN" in os.environ and type(os.environ["AZURE_TOKEN"]) == str:
    print('AZURE_TOKEN_LEN', len(os.environ["AZURE_TOKEN"]))

  sas_token = None
  if os.path.isfile(TOKEN_PATH):
    sas_token = open(TOKEN_PATH).read().strip()
  elif "AZURE_TOKEN" in os.environ:
    sas_token = os.environ["AZURE_TOKEN"]

  if sas_token is None:
    sas_token = subprocess.check_output("az storage container generate-sas --account-name commadataci --name openpilotci --https-only --permissions lrw \
                                         --expiry $(date -u '+%Y-%m-%dT%H:%M:%SZ' -d '+1 hour') --auth-mode login --as-user --output tsv", shell=True).decode().strip("\n")
  service = BlockBlobService(account_name="commadataci", sas_token=sas_token)
  service.create_blob_from_path("openpilotci", name, path)
  return "https://commadataci.blob.core.windows.net/openpilotci/" + name


if __name__ == "__main__":
  for f in sys.argv[1:]:
    name = os.path.basename(f)
    url = upload_file(f, name)
    print(url)
