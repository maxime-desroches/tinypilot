#!/usr/bin/env python3
import argparse
import bz2
import zstd
import lz4.frame
import zlib
from collections import defaultdict

import matplotlib.pyplot as plt

from cereal.services import SERVICE_LIST
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route
from tqdm import tqdm

MIN_SIZE = 0.5  # Percent size of total to show as separate entry


def compress(data):
  # return lz4.frame.compress(data)
  # return zlib.compress(data)
  # return bz2.compress(data)
  return zstd.compress(data)


def make_pie(msgs, typ):
  msgs_by_type = defaultdict(list)
  print(len(msgs))
  for m in msgs:
    # # print(dir(m.as_builder()))
    # print(m.which())
    # print(len(m.as_builder().to_bytes()))
    # print(len(m.as_builder().to_bytes_packed()))
    # break
    msgs_by_type[m.which()].append(m.as_builder().to_bytes_packed())

  length_by_type = {k: len(b"".join(v)) for k, v in msgs_by_type.items()}
  compressed_length_by_type = {k: len(compress(b"".join(v))) for k, v in msgs_by_type.items()}

  compressed_length_by_type_v2 = {}

  total = sum(compressed_length_by_type.values())
  real_total = len(compress(b"".join([m.as_builder().to_bytes_packed() for m in msgs])))
  uncompressed_total = len(b"".join([m.as_builder().to_bytes_packed() for m in msgs]))

  for k in tqdm(msgs_by_type.keys()):
    compressed_length_by_type_v2[k] = real_total - len(compress(b"".join([m.as_builder().to_bytes_packed() for m in msgs if m.which() != k])))
    # print(k, compressed_length_by_type_v2[k])

  sizes = sorted(compressed_length_by_type.items(), key=lambda kv: kv[1])
  # sizes = sorted(compressed_length_by_type_v2.items(), key=lambda kv: kv[1])

  print("name - comp. size (uncomp. size)")
  for (name, sz) in sizes:
    print(f"{name:<22} - {sz / 1024:.2f} kB / {compressed_length_by_type_v2[name] / 1024:.2f} kB ({length_by_type[name] / 1024:.2f} kB)")
  print()
  print(f"{typ} - Total {total / 1024:.2f} kB")
  print(f"{typ} - Real {real_total / 1024:.2f} kB")
  print(f"{typ} - Simulated total (v2) {sum(compressed_length_by_type_v2.values()) / 1024:.2f} MB")
  print(f"{typ} - Uncompressed total {uncompressed_total / 1024 / 1024:.2f} MB")

  sizes_large = [(k, sz) for (k, sz) in sizes if sz >= total * MIN_SIZE / 100]
  sizes_large += [('other', sum(sz for (_, sz) in sizes if sz < total * MIN_SIZE / 100))]

  labels, sizes = zip(*sizes_large, strict=True)

  plt.figure()
  plt.title(f"{typ}")
  plt.pie(sizes, labels=labels, autopct='%1.1f%%')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Check qlog size based on a rlog')
  parser.add_argument('route', help='route to use')
  args = parser.parse_args()

  msgs = list(LogReader(args.route))

  make_pie(msgs, 'qlog')
  plt.show()
