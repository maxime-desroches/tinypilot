#!/usr/bin/env python

import argparse
import pathlib

from openpilot.system.updated.casync.common import create_caexclude_file, create_casync_channel, create_manifest_file


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="creates a casync release")
  parser.add_argument("target_dir", type=str, help="target directory to build channel from")
  parser.add_argument("output_dir", type=str, help="output directory for the channel")
  parser.add_argument("channel", type=str, help="what channel this build is")
  args = parser.parse_args()

  target_dir = pathlib.Path(args.target_dir)
  output_dir = pathlib.Path(args.output_dir)

  create_manifest_file(target_dir, args.channel)
  create_caexclude_file(target_dir)

  digest, caidx = create_casync_channel(target_dir, output_dir, args.channel)

  print(f"Created casync channel from {target_dir} to {caidx} with digest {digest}")
