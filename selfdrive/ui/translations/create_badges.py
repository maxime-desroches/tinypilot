#!/usr/bin/env python3
import json
import os
import requests

from common.basedir import BASEDIR
from selfdrive.ui.update_translations import LANGUAGES_FILE, TRANSLATIONS_DIR

TRANSLATION_TAG = "<translation"
UNFINISHED_TRANSLATION_TAG = "<translation type=\"unfinished\""
BADGE_HEIGHT = 20
BADGE_OFFSET = 8

if __name__ == "__main__":
  with open(LANGUAGES_FILE, "r") as f:
    translation_files = json.load(f)

  global_svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="{len(translation_files) * (BADGE_HEIGHT + BADGE_OFFSET)}">']
  for idx, (name, file) in enumerate(translation_files.items()):
    with open(os.path.join(TRANSLATIONS_DIR, f"{file}.ts"), "r") as tr_f:
      tr_file = tr_f.read()

    total_translations = 0
    unfinished_translations = 0
    for line in tr_file.splitlines():
      if TRANSLATION_TAG in line:
        total_translations += 1
      if UNFINISHED_TRANSLATION_TAG in line:
        unfinished_translations += 1

    percent_finished = int(100 - (unfinished_translations / total_translations * 100.))
    color = "green" if percent_finished == 100 else "orange" if percent_finished >= 70 else "red"

    r = requests.get(f"https://img.shields.io/badge/LANGUAGE {name}-{percent_finished}%25 complete-{color}")
    assert r.status_code == 200, "Error downloading badge"
    content_svg = r.content.decode("utf-8")

    # need to make tag ids in each badge unique
    for tag in ("r", "s"):
      content_svg = content_svg.replace(f'id="{tag}"', f'id="{tag}{idx}"')
      content_svg = content_svg.replace(f'"url(#{tag})"', f'"url(#{tag}{idx})"')

    global_svg.append('<g transform="translate(0, {})">'.format(idx * (BADGE_HEIGHT + BADGE_OFFSET)))
    global_svg.append(content_svg)
    global_svg.append("</g>")

  global_svg.append("</svg>")

  with open(os.path.join(BASEDIR, "translation_badge.svg"), "w") as badge_f:
    badge_f.write("\n".join(global_svg))
