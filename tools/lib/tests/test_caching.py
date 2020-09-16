#!/usr/bin/env python3

import unittest
from tools.lib.url_file import URLFile

class TestFileDownload(unittest.TestCase):

  def test_downloads(self):
    self.assertEqual(1+1,2)
    small_file_url = "https://raw.githubusercontent.com/commaai/openpilot/master/SAFETY.md"

    #Load the end 100 bytes of both files
    file_small_cached2 = URLFile(small_file_url)
    file_small_download2 = URLFile(small_file_url, download=True)
    self.assertEqual(file_small_cached2.get_length(),file_small_download2.get_length())
    length = file_small_download2.get_length()
    #print(length)
    file_small_cached2.seek(length-100)
    file_small_download2.seek(length-100)
    self.assertEqual(file_small_cached2.read(),file_small_download2.read())
    
    #Load full small file
    file_small_cached = URLFile(small_file_url)
    file_small_download = URLFile(small_file_url, download=True)
    self.assertEqual(file_small_cached.get_length(),file_small_download.get_length())
    self.assertEqual(file_small_cached.read(),file_small_download.read())

    large_file_url = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/fcamera.hevc"
    
    #Load the end 100 bytes of both files
    file_large_cached2 = URLFile(large_file_url)
    file_large_download2 = URLFile(large_file_url, download=True)
    #print(file_large_cached2.get_length())
    #print(file_large_download2.get_length())
    self.assertEqual(file_large_cached2.get_length(),file_large_download2.get_length())
    length = file_large_download2.get_length()
    file_large_download2.seek(length-100)
    file_large_cached2.seek(length-100)
    self.assertEqual(file_large_cached2.read(),file_large_download2.read())

    #Load full large file
    file_large_cached = URLFile(large_file_url)
    file_large_download = URLFile(large_file_url, download=True)
    self.assertEqual(file_large_cached.get_length(),file_large_download.get_length())
    self.assertEqual(file_large_cached.read(),file_large_download.read())

    




if __name__ == "__main__":
  unittest.main()
