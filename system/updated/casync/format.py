import abc
from dataclasses import dataclass
import io
import struct

from openpilot.system.updated.casync.chunk_reader import CAChunk


# from https://github.com/systemd/casync/blob/e6817a79d89b48e1c6083fb1868a28f1afb32505/src/caformat.h#L49

FLAG_PERMISSIONS = 0xb000000000000100
FLAG_SYMLINKS = 0xb000000000000200

REQUIRED_FLAGS = FLAG_SYMLINKS | FLAG_PERMISSIONS # correspond to [--with=symlinks --with=permissions]

CA_HEADER_LEN = 32
CA_TABLE_HEADER_LEN = 16
CA_TABLE_ENTRY_LEN = 40
CA_TABLE_MIN_LEN = CA_TABLE_HEADER_LEN + CA_TABLE_ENTRY_LEN

CA_FORMAT_INDEX = 0x96824d9c7b129ff9
CA_FORMAT_ENTRY = 0x1396fabcea5bbb51
CA_FORMAT_GOODBYE = 0xdfd35c5e8327c403
CA_FORMAT_FILENAME = 0x6dbb6ebcb3161f0b
CA_FORMAT_PAYLOAD = 0x8b9e1d93d6dcffc9
CA_FORMAT_SYMLINK = 0x664a6fb6830e0d6c

CA_FORMAT_TABLE = 0xe75b9e112f17417d
CA_FORMAT_TABLE_TAIL_MARKER = 0x4b4f050e5549ecd1

CA_MAX_FILENAME_SIZE = 256


@dataclass
class CAFormatHeader(abc.ABC):
  size: int
  format_type: int

  @staticmethod
  def from_buffer(b: io.BytesIO) -> 'CAFormatHeader':
    size, format_type = struct.unpack("<QQ", b.read(CA_TABLE_HEADER_LEN))

    return CAFormatHeader(size, format_type)


@dataclass
class CaFormatIndex:
  header: CAFormatHeader
  flags: int
  chunk_size_min: int
  chunk_size_avg: int
  chunk_size_max: int

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)
    assert header.format_type == CA_FORMAT_INDEX
    return CaFormatIndex(header, *struct.unpack("<QQQQ", b.read(CA_HEADER_LEN)))


@dataclass
class CATableItem:
  offset: int
  chunk_id: bytes

  marker: int

  @staticmethod
  def from_buffer(b: io.BytesIO):
    offset = struct.unpack("<Q", b.read(8))[0]
    chunk_id = b.read(32)

    marker = struct.unpack("<Q", chunk_id[24:])[0]

    return CATableItem(offset, chunk_id, marker)


@dataclass
class CATable:
  index: CaFormatIndex

  items: list[CATableItem]

  @staticmethod
  def from_buffer(b: io.BytesIO):
    index = CaFormatIndex.from_buffer(b)
    header = CAFormatHeader.from_buffer(b)
    assert header.format_type == CA_FORMAT_TABLE

    items = []
    while True:
      item = CATableItem.from_buffer(b)

      if item.marker == CA_FORMAT_TABLE_TAIL_MARKER:
        break
      items.append(item)

    return CATable(index, items)

  def to_chunks(self) -> list[CAChunk]:
    offset = 0
    ret = []
    for i, item in enumerate(self.items):
      length = item.offset - offset

      assert length <= self.index.chunk_size_max

      # Last chunk can be smaller
      if i < len(self.items) - 1:
        assert length >= self.index.chunk_size_min

      ret.append(CAChunk(item.chunk_id, offset, length))
      offset += length

    return ret


@dataclass
class CAIndex:
  chunks: list[CAChunk]

  @staticmethod
  def from_buffer(b: io.BytesIO):
    table = CATable.from_buffer(b)

    chunks = table.to_chunks()

    return CAIndex(chunks)

  @staticmethod
  def from_file(filepath):
    with open(filepath, "rb") as f:
      return CAIndex.from_buffer(f)


@dataclass
class CAFilename:
  filename: str | None

  @staticmethod
  def from_buffer(b: io.BytesIO):
    _ = CAFormatHeader.from_buffer(b)

    filename = b""

    while len(filename) < CA_MAX_FILENAME_SIZE:
      c = b.read(1)
      if c == b'\x00':
        break
      filename += c

    return CAFilename(filename.decode("utf-8"))


@dataclass
class CAEntry:
  feature_flags: int
  mode: int
  flags: int
  uid: int
  gid: int
  mtime: int

  @staticmethod
  def from_buffer(b: io.BytesIO):
    _ = CAFormatHeader.from_buffer(b)
    return CAEntry(*struct.unpack("<QQQQQQ", b.read(8*6)))


@dataclass
class CAPayload:
  data: bytes

  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)

    data = b.read(header.size - 16)
    return CAPayload(data)

  def __repr__(self):
    return f"CAPayload(data={len(self.data)} bytes)"


@dataclass
class CASymlink:
  target: str

  @staticmethod
  def from_buffer(b: io.BytesIO):
    _ = CAFormatHeader.from_buffer(b)

    target = b""

    while len(target) < CA_MAX_FILENAME_SIZE:
      c = b.read(1)
      if c == b'\x00':
        break
      target += c

    return CASymlink(target.decode("utf-8"))


@dataclass
class CAGoodbye:
  @staticmethod
  def from_buffer(b: io.BytesIO):
    header = CAFormatHeader.from_buffer(b)
    b.read(header.size - 16)

    return CAGoodbye()


@dataclass
class CAArchive:
  items: list

  @staticmethod
  def from_buffer(b: io.BytesIO) -> 'CAArchive':
    entry = CAEntry.from_buffer(b)

    assert entry.feature_flags == REQUIRED_FLAGS
    items = [entry]

    cur = b.tell()

    size = b.seek(0, 2)
    b.seek(cur, 0)

    while b.tell() < size:
      header = CAFormatHeader.from_buffer(b)
      b.seek(-16, 1) # reset back to header

      if header.format_type == CA_FORMAT_FILENAME:
        filename = CAFilename.from_buffer(b)
        archive = CAArchive.from_buffer(b)
        items.append(filename)
        items.append(archive)
      elif header.format_type == CA_FORMAT_GOODBYE:
        goodbye = CAGoodbye.from_buffer(b)
        items.append(goodbye)
        break
      elif header.format_type == CA_FORMAT_PAYLOAD:
        payload = CAPayload.from_buffer(b)
        items.append(payload)
        break
      elif header.format_type == CA_FORMAT_SYMLINK:
        symlink = CASymlink.from_buffer(b)
        items.append(symlink)
        break
      else:
        raise Exception(f"unsupported type: {header.format_type:02x}")

    return CAArchive(items)
