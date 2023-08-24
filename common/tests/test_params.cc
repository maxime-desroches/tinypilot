
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#define private public
#include "common/params.h"

TEST_CASE("Params/asyncWriter") {
  char tmp_path[] = "/tmp/asyncWriter_XXXXXX";
  const std::string param_path = mkdtemp(tmp_path);
  Params params(param_path);

  AsyncWriter async_writer;
  auto param_names = {"CarParams", "IsMetric"};
  for (const auto &name : param_names) {
    async_writer.queue({param_path, name, "1"});
    // param is empty
    REQUIRE(params.get(name).empty());
  }

  // check if thread is running
  REQUIRE(async_writer.future.valid());
  REQUIRE(async_writer.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);

  // wait for finish
  async_writer.future.wait();
  REQUIRE(async_writer.q.size() == 0);

  // check results
  for (const auto &name : param_names) {
    REQUIRE(params.get(name) == "1");
  }
}
