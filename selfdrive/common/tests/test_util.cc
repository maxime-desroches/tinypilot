
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <climits>
#include <random>
#include <string>

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"

std::string random_bytes(int size) {
  std::random_device rd;
  std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe(rd());
  std::string bytes(size+1, '\0');
  std::generate(bytes.begin(), bytes.end(), std::ref(rbe));
  return bytes;
}

TEST_CASE("util::read_file") {
  SECTION("read /proc/version") {
    std::string ret = util::read_file("/proc/version");
    REQUIRE(ret.find("Linux version") != std::string::npos);
  }
  SECTION("read from sysfs") {
    std::string ret = util::read_file("/sys/power/wakeup_count");
    REQUIRE(!ret.empty());
  }
  SECTION("read file") {
    char filename[] = "/tmp/test_read_XXXXXX";
    int fd = mkstemp(filename);

    REQUIRE(util::read_file(filename).empty());

    std::string content = random_bytes(64 * 1024);
    write(fd, content.c_str(), content.size());
    std::string ret = util::read_file(filename);
    REQUIRE(ret == content);
    close(fd);
  }
  SECTION("read directory") {
    REQUIRE(util::read_file(".").empty());
  }
  SECTION("read non-existent file") {
    std::string ret = util::read_file("does_not_exist");
    REQUIRE(ret.empty());
  }
  SECTION("read non-permission") {
    REQUIRE(util::read_file("/proc/kmsg").empty());
  }
}

TEST_CASE("util::file_exists") {
  const std::string file = "/tmp/test_util_file_exists";
  system(("rm " + file + " -f").c_str());
  int flock = ::open(file.c_str(), O_RDWR | O_CREAT);
  assert(flock != -1);
  REQUIRE(util::file_exists(file) == true);
  close(flock);
}
