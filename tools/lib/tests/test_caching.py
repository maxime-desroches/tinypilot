#!/usr/bin/env python3
import http.server
import os
from parameterized import parameterized
import threading
import unittest
from openpilot.tools.lib.url_file import URLFile


class MockRequestHandler(http.server.BaseHTTPRequestHandler):
  FILE_EXISTS = True

  def _handle_request(self, send_content):
    if self.FILE_EXISTS:
      self.send_response(200)
      self.send_header("Content-Length", 4)
    else:
      self.send_response(404)
    self.end_headers()

    if self.FILE_EXISTS and send_content:
      self.wfile.write(b'1234')

  def do_GET(self):
    self._handle_request(True)

  def do_HEAD(self):
    self._handle_request(False)


class MockServer(threading.Thread):
  def run(self):
    self.server = http.server.HTTPServer(("127.0.0.1", 5001), MockRequestHandler)
    self.server.serve_forever()

  def stop(self):
    self.server.shutdown()


class TestFileDownload(unittest.TestCase):
  server: MockServer

  @classmethod
  def setUpClass(cls):
    cls.server = MockServer()
    cls.server.start()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  def compare_loads(self, url, start=0, length=None):
    """Compares range between cached and non cached version"""
    file_cached = URLFile(url, cache=True)
    file_downloaded = URLFile(url, cache=False)

    file_cached.seek(start)
    file_downloaded.seek(start)

    self.assertEqual(file_cached.get_length(), file_downloaded.get_length())
    self.assertLessEqual(length + start if length is not None else 0, file_downloaded.get_length())

    response_cached = file_cached.read(ll=length)
    response_downloaded = file_downloaded.read(ll=length)

    self.assertEqual(response_cached, response_downloaded)

    # Now test with cache in place
    file_cached = URLFile(url, cache=True)
    file_cached.seek(start)
    response_cached = file_cached.read(ll=length)

    self.assertEqual(file_cached.get_length(), file_downloaded.get_length())
    self.assertEqual(response_cached, response_downloaded)

  def test_small_file(self):
    # Make sure we don't force cache
    os.environ["FILEREADER_CACHE"] = "0"
    small_file_url = "https://raw.githubusercontent.com/commaai/openpilot/master/docs/SAFETY.md"
    #  If you want large file to be larger than a chunk
    #  large_file_url = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/fcamera.hevc"

    #  Load full small file
    self.compare_loads(small_file_url)

    file_small = URLFile(small_file_url)
    length = file_small.get_length()

    self.compare_loads(small_file_url, length - 100, 100)
    self.compare_loads(small_file_url, 50, 100)

    #  Load small file 100 bytes at a time
    for i in range(length // 100):
      self.compare_loads(small_file_url, 100 * i, 100)

  def test_large_file(self):
    large_file_url = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/qlog.bz2"
    #  Load the end 100 bytes of both files
    file_large = URLFile(large_file_url)
    length = file_large.get_length()

    self.compare_loads(large_file_url, length - 100, 100)
    self.compare_loads(large_file_url)

  @parameterized.expand([(True, ), (False, )])
  def test_exists_file(self, cache_enabled):
    os.environ["FILEREADER_CACHE"] = "1" if cache_enabled else "0"

    exists_file = "http://localhost:5001/test.png"

    MockRequestHandler.FILE_EXISTS = False
    file_not_exists = URLFile(exists_file)
    length = file_not_exists.get_length()
    self.assertEqual(length, -1)

    MockRequestHandler.FILE_EXISTS = True
    file_exists = URLFile(exists_file)
    length = file_exists.get_length()
    self.assertEqual(length, 4)


if __name__ == "__main__":
  unittest.main()
