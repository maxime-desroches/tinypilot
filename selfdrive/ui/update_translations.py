#!/usr/bin/env python3
import argparse
import json
import os

from openpilot.common.basedir import BASEDIR

UI_DIR = os.path.join(BASEDIR, "selfdrive", "ui")
TRANSLATIONS_DIR = os.path.join(UI_DIR, "translations")
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, "languages.json")
TRANSLATIONS_INCLUDE_FILE = os.path.join(TRANSLATIONS_DIR, "alerts_generated.h")
PLURAL_ONLY = ["main_en"]  # base language, only create entries for strings with plural forms


def generate_translations_include():
  # offroad alerts
  # TODO translate events from openpilot.selfdrive/controls/lib/events.py
  content = "// THIS IS AN AUTOGENERATED FILE, PLEASE EDIT alerts_offroad.json\n"
  with open(os.path.join(BASEDIR, "selfdrive/controls/lib/alerts_offroad.json")) as f:
    for alert in json.load(f).values():
      content += f'QT_TRANSLATE_NOOP("OffroadAlert", R"({alert["text"]})");\n'

  with open(TRANSLATIONS_INCLUDE_FILE, "w") as f:
    f.write(content)


def update_translations(vanish: bool = False, translation_files: None | list[str] = None, translations_dir: str = TRANSLATIONS_DIR):
  generate_translations_include()

  if translation_files is None:
    with open(LANGUAGES_FILE, "r") as f:
      translation_files = json.load(f).values()

  for file in translation_files:
    tr_file = os.path.join(translations_dir, f"{file}.ts")
    args = f"lupdate -locations none -recursive {UI_DIR} -ts {tr_file} -I {BASEDIR}"
    if vanish:
      args += " -no-obsolete"
    if file in PLURAL_ONLY:
      args += " -pluralonly"
    ret = os.system(args)
    assert ret == 0


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Update translation files for UI",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--vanish", action="store_true", help="Remove translations with source text no longer found")
  args = parser.parse_args()

  update_translations(args.vanish)
