import gc
import os
import pytest
import random
import subprocess

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.manager import manager
from openpilot.system.hardware import PC, TICI, HARDWARE


def pytest_sessionstart(session):
  # TODO: fix tests and enable test order randomization
  if session.config.pluginmanager.hasplugin('randomly'):
    session.config.option.randomly_reorganize = False


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_call(item):
  # ensure we run as a hook after capturemanager's
  if item.get_closest_marker("nocapture") is not None:
    capmanager = item.config.pluginmanager.getplugin("capturemanager")
    with capmanager.global_and_fixture_disabled():
      yield
  else:
    yield


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture(request):
  starting_env = dict(os.environ)

  random.seed(0)

  # setup a clean environment for each test
  with OpenpilotPrefix(shared_download_cache=request.node.get_closest_marker("shared_download_cache") is not None) as prefix:
    prefix = os.environ["OPENPILOT_PREFIX"]

    yield

    # ensure the test doesn't change the prefix
    assert "OPENPILOT_PREFIX" in os.environ and prefix == os.environ["OPENPILOT_PREFIX"]

  os.environ.clear()
  os.environ.update(starting_env)

  # cleanup any started processes
  manager.manager_cleanup()

  # some processes disable gc for performance, re-enable here
  if not gc.isenabled():
    gc.enable()
    gc.collect()

# If you use setUpClass, the environment variables won't be cleared properly,
# so we need to hook both the function and class pytest fixtures
@pytest.fixture(scope="class", autouse=True)
def openpilot_class_fixture():
  starting_env = dict(os.environ)

  yield

  os.environ.clear()
  os.environ.update(starting_env)


@pytest.fixture(scope="function")
def tici_setup_fixture(openpilot_function_fixture):
  """Ensure a consistent state for tests on-device. Needs the openpilot function fixture to run first."""
  HARDWARE.initialize_hardware()
  HARDWARE.set_power_save(False)
  os.system("pkill -9 -f athena")


def process_running(process_name):
  ret = subprocess.run(["pgrep", "-f", f"\\/{process_name}[^\\/]"]).returncode
  return ret == 0


@pytest.fixture(scope="session", autouse=True)
def tici_session_fixture():
  """Ensure that the device has a valid environment to run tests, to better highlight cleanup failures."""

  if PC:
    yield
    return # pc tests are isolated with the prefix

  def ensure_no_processes():
    still_running = [proc for proc in manager.managed_processes if (process_running(proc) and proc not in ["manage_athenad"])]
    assert not still_running, f"Processes still running: {still_running}"

  ensure_no_processes()
  yield
  ensure_no_processes()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
  skipper = pytest.mark.skip(reason="Skipping tici test on PC")
  for item in items:
    if "tici" in item.keywords:
      if not TICI:
        item.add_marker(skipper)
      else:
        item.fixturenames.append('tici_setup_fixture')

    if "xdist_group_class_property" in item.keywords:
      class_property_name = item.get_closest_marker('xdist_group_class_property').args[0]
      class_property_value = getattr(item.cls, class_property_name)
      item.add_marker(pytest.mark.xdist_group(class_property_value))


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
  config_line = "xdist_group_class_property: group tests by a property of the class that contains them"
  config.addinivalue_line("markers", config_line)

  config_line = "nocapture: don't capture test output"
  config.addinivalue_line("markers", config_line)

  config_line = "shared_download_cache: share download cache between tests"
  config.addinivalue_line("markers", config_line)
