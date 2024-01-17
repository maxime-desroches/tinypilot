import os

import requests
from openpilot.tools.lib.url_file import URLFile

DATA_ENDPOINT = os.getenv("DATA_ENDPOINT", "http://data-raw.comma.internal/")

def resolve_name(fn):
  if fn.startswith("cd:/"):
    return fn.replace("cd:/", DATA_ENDPOINT)
  return fn

def file_exists(fn):
  fn = resolve_name(fn)
  if fn.startswith(("http://", "https://")):
    return requests.head(fn, allow_redirects=True).status_code == 200
  return os.path.exists(fn)

def FileReader(fn, debug=False):
  fn = resolve_name(fn)
  if fn.startswith(("http://", "https://")):
    return URLFile(fn, debug=debug)
  return open(fn, "rb")
