import time
from functools import wraps

from selfdrive.version import training_version, terms_version
from selfdrive.manager.process_config import managed_processes


def set_params_enabled():
  from common.params import Params
  params = Params()
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put_bool("OpenpilotEnabledToggle", True)
  params.put_bool("CommunityFeaturesToggle", True)
  params.put_bool("Passive", False)


# TODO: rewrite for unittest
#def phone_only(x):
#  if PC:
#    return nottest(x)
#  else:
#    return x


def with_processes(processes, init_time=0, ignore_stopped=None):
  ignore_stopped = [] if ignore_stopped is None else ignore_stopped

  def wrapper(func):
    @wraps(func)
    def wrap(*args, **kwargs):
      # start and assert started
      for n, p in enumerate(processes):
        managed_processes[p].start()
        if n < len(processes) - 1:
          time.sleep(init_time)
      assert all(managed_processes[name].proc.exitcode is None for name in processes)

      # call the function
      try:
        func(*args, **kwargs)
        # assert processes are still started
        assert all(managed_processes[name].proc.exitcode is None for name in processes if name not in ignore_stopped)
      finally:
        for p in processes:
          managed_processes[p].stop()

    return wrap
  return wrapper
