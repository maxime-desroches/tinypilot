import sys
import unittest

OUTPUT_FILE = sys.argv[1]
WORKSPACE = sys.argv[2] 

class Test_RelCached(unittest.TestCase):

  def test_relcache(self):
    valid_start_strings = ['Compiling /', '[1/1] Cythonizing']
    with open(OUTPUT_FILE, 'r') as f:
      for line in f:
        if line.startswith(valid_start_strings[0]) or line.startswith(valid_start_strings[1]):
          continue
        else:
          words = line.split(" ")
          for wrd in words:
            if wrd == WORKSPACE:
              self.fail("Contains absolute path while building")

if __name__ == "__main__":
  del sys.argv[1:]
  unittest.main()
