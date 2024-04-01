from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
import io
import lzma
import os
import struct
import time

import requests


CHUNK_DOWNLOAD_TIMEOUT = 60
CHUNK_DOWNLOAD_RETRIES = 3

CAIBX_DOWNLOAD_TIMEOUT = 120


@dataclass(unsafe_hash=True)
class CAChunk:
  sha: bytes
  offset: int
  length: int

  @staticmethod
  def from_buffer(b: io.BytesIO, last_offset: int):
    new_offset = struct.unpack("<Q", b.read(8))[0]

    sha = b.read(32)
    length = new_offset - last_offset

    return CAChunk(sha, last_offset, length)


class ChunkReader(ABC):
  @abstractmethod
  def read(self, chunk: CAChunk) -> bytes:
    ...


class ChunkBuffer(io.BytesIO):
  """Reads a list of chunks as a contiguous stream"""
  def __init__(self, chunks: list[CAChunk], chunk_reader: ChunkReader):
    self.chunks = chunks
    self.offset = 0
    self.chunk_reader = chunk_reader

    self.chunk_cache: OrderedDict[CAChunk, bytes] = OrderedDict()

  @property
  def length(self):
    return sum(chunk.length for chunk in self.chunks)

  def get_current_chunk_and_offset(self):
    i = 0
    size = 0
    while size + self.chunks[i].length - 1 < self.offset:
      size += self.chunks[i].length
      i += 1
    return self.chunks[i], self.offset - size

  def read(self, size: int | None = None) -> bytes:
    ret = b""
    while size is not None and size > 0:
      current_chunk, chunk_offset = self.get_current_chunk_and_offset()
      if current_chunk not in self.chunk_cache:
        self.chunk_cache[current_chunk] = self.chunk_reader.read(current_chunk)

      to_read = min(size, current_chunk.length - chunk_offset)
      ret += self.chunk_cache[current_chunk][chunk_offset:chunk_offset+to_read]
      size -= to_read

      self.offset += to_read

      if len(self.chunk_cache) > 1024:
        self.chunk_cache.popitem(last=False)

    return ret

  def seek(self, offset, whence):
    if whence == 0:
      self.offset = offset
    if whence == 1:
      self.offset += offset
    if whence == 2:
      self.offset = self.length + self.offset

    return self.tell()

  def tell(self):
    return self.offset


class FileChunkReader(ChunkReader):
  """Reads chunks from a local file"""
  def __init__(self, fn: str) -> None:
    super().__init__()
    self.f = open(fn, 'rb')

  def __del__(self):
    self.f.close()

  def read(self, chunk: CAChunk) -> bytes:
    self.f.seek(chunk.offset)
    return self.f.read(chunk.length)


class RemoteChunkReader(ChunkReader):
  """Reads lzma compressed chunks from a remote store"""

  def __init__(self, url: str) -> None:
    super().__init__()
    self.url = url
    self.session = requests.Session()

  def read(self, chunk: CAChunk) -> bytes:
    sha_hex = chunk.sha.hex()
    url = os.path.join(self.url, sha_hex[:4], sha_hex + ".cacnk")

    if os.path.isfile(url):
      with open(url, 'rb') as f:
        contents = f.read()
    else:
      for i in range(CHUNK_DOWNLOAD_RETRIES):
        try:
          resp = self.session.get(url, timeout=CHUNK_DOWNLOAD_TIMEOUT)
          break
        except Exception:
          if i == CHUNK_DOWNLOAD_RETRIES - 1:
            raise
          time.sleep(CHUNK_DOWNLOAD_TIMEOUT)

      resp.raise_for_status()
      contents = resp.content

    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
    return decompressor.decompress(contents)


class DirectoryChunkReader(ChunkReader):
  """Reads chunks from a local file"""
  def __init__(self, directory: str) -> None:
    super().__init__()
    self.directory = directory

  def read(self, chunk: CAChunk) -> bytes:
    sha_hex = chunk.sha.hex()
    filename = os.path.join(self.directory, sha_hex[:4], sha_hex + ".cacnk")

    with open(filename, "rb") as f:
      decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
      return decompressor.decompress(f.read())


def AutoChunkReader(path: str):
  if "http" in path:
    return RemoteChunkReader(path)
  else:
    return DirectoryChunkReader(path)
