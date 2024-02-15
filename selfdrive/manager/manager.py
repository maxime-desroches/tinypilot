#!/usr/bin/env python3
import datetime
import os
import signal
import sys
import traceback
from typing import List, Tuple, Union

from cereal import log
import cereal.messaging as messaging
import openpilot.selfdrive.sentry as sentry
from openpilot.common.params import Params, ParamKeyType
from openpilot.common.text_window import TextWindow
from openpilot.system.hardware import HARDWARE, PC
from openpilot.selfdrive.manager.helpers import unblock_stdout, write_onroad_params, save_bootlog
from openpilot.selfdrive.manager.process import ensure_running
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.selfdrive.athena.registration import register, UNREGISTERED_DONGLE_ID
from openpilot.common.swaglog import cloudlog, add_file_handler
from openpilot.system.version import is_dirty, get_commit, get_version, get_origin, get_short_branch, \
                           get_normalized_origin, terms_version, training_version, \
                           is_tested_branch, is_release_branch



class Manager:
  BOOTLOG_THREAD = None
  
  @staticmethod
  def init() -> None:
    Manager.BOOTLOG_THREAD = save_bootlog()

    params = Params()
    params.clear_all(ParamKeyType.CLEAR_ON_MANAGER_START)
    params.clear_all(ParamKeyType.CLEAR_ON_ONROAD_TRANSITION)
    params.clear_all(ParamKeyType.CLEAR_ON_OFFROAD_TRANSITION)
    if is_release_branch():
      params.clear_all(ParamKeyType.DEVELOPMENT_ONLY)

    default_params: List[Tuple[str, Union[str, bytes]]] = [
      ("CompletedTrainingVersion", "0"),
      ("DisengageOnAccelerator", "0"),
      ("GsmMetered", "1"),
      ("HasAcceptedTerms", "0"),
      ("LanguageSetting", "main_en"),
      ("OpenpilotEnabledToggle", "1"),
      ("LongitudinalPersonality", str(log.LongitudinalPersonality.standard)),
    ]
    if not PC:
      default_params.append(("LastUpdateTime", datetime.datetime.utcnow().isoformat().encode('utf8')))

    if params.get_bool("RecordFrontLock"):
      params.put_bool("RecordFront", True)

    # set unset params
    for k, v in default_params:
      if params.get(k) is None:
        params.put(k, v)

    # Create folders needed for msgq
    try:
      os.mkdir("/dev/shm")
    except FileExistsError:
      pass
    except PermissionError:
      print("WARNING: failed to make /dev/shm")

    # set version params
    params.put("Version", get_version())
    params.put("TermsVersion", terms_version)
    params.put("TrainingVersion", training_version)
    params.put("GitCommit", get_commit())
    params.put("GitBranch", get_short_branch())
    params.put("GitRemote", get_origin())
    params.put_bool("IsTestedBranch", is_tested_branch())
    params.put_bool("IsReleaseBranch", is_release_branch())

    # set dongle id
    reg_res = register(show_spinner=True)
    if reg_res:
      dongle_id = reg_res
    else:
      serial = params.get("HardwareSerial")
      raise Exception(f"Registration failed for device {serial}")
    os.environ['DONGLE_ID'] = dongle_id  # Needed for swaglog
    os.environ['GIT_ORIGIN'] = get_normalized_origin() # Needed for swaglog
    os.environ['GIT_BRANCH'] = get_short_branch() # Needed for swaglog
    os.environ['GIT_COMMIT'] = get_commit() # Needed for swaglog

    if not is_dirty():
      os.environ['CLEAN'] = '1'

    # init logging
    sentry.init(sentry.SentryProject.SELFDRIVE)
    cloudlog.bind_global(dongle_id=dongle_id,
                        version=get_version(),
                        origin=get_normalized_origin(),
                        branch=get_short_branch(),
                        commit=get_commit(),
                        dirty=is_dirty(),
                        device=HARDWARE.get_device_type())

    # preimport all processes
    for p in managed_processes.values():
      p.prepare()

  @staticmethod
  def cleanup() -> None:
    # send signals to kill all procs
    for p in managed_processes.values():
      p.stop(block=False)

    # ensure all are killed
    for p in managed_processes.values():
      p.stop(block=True)

    if Manager.BOOTLOG_THREAD is not None:
      Manager.BOOTLOG_THREAD.join()

    cloudlog.info("everything is dead")


def manager_init() -> None:
  return Manager.init()


def manager_cleanup() -> None:
  return Manager.cleanup()


def manager_thread() -> None:
  cloudlog.bind(daemon="manager")
  cloudlog.info("manager start")
  cloudlog.info({"environ": os.environ})

  params = Params()

  ignore: List[str] = []
  if params.get("DongleId", encoding='utf8') in (None, UNREGISTERED_DONGLE_ID):
    ignore += ["manage_athenad", "uploader"]
  if os.getenv("NOBOARD") is not None:
    ignore.append("pandad")
  ignore += [x for x in os.getenv("BLOCK", "").split(",") if len(x) > 0]

  sm = messaging.SubMaster(['deviceState', 'carParams'], poll='deviceState')
  pm = messaging.PubMaster(['managerState'])

  write_onroad_params(False, params)
  ensure_running(managed_processes.values(), False, params=params, CP=sm['carParams'], not_run=ignore)

  started_prev = False

  while True:
    sm.update(1000)

    started = sm['deviceState'].started

    if started and not started_prev:
      params.clear_all(ParamKeyType.CLEAR_ON_ONROAD_TRANSITION)
    elif not started and started_prev:
      params.clear_all(ParamKeyType.CLEAR_ON_OFFROAD_TRANSITION)

    # update onroad params, which drives boardd's safety setter thread
    if started != started_prev:
      write_onroad_params(started, params)

    started_prev = started

    ensure_running(managed_processes.values(), started, params=params, CP=sm['carParams'], not_run=ignore)

    running = ' '.join("%s%s\u001b[0m" % ("\u001b[32m" if p.proc.is_alive() else "\u001b[31m", p.name)
                       for p in managed_processes.values() if p.proc)
    print(running)
    cloudlog.debug(running)

    # send managerState
    msg = messaging.new_message('managerState', valid=True)
    msg.managerState.processes = [p.get_process_state_msg() for p in managed_processes.values()]
    pm.send('managerState', msg)

    # Exit main loop when uninstall/shutdown/reboot is needed
    shutdown = False
    for param in ("DoUninstall", "DoShutdown", "DoReboot"):
      if params.get_bool(param):
        shutdown = True
        params.put("LastManagerExitReason", f"{param} {datetime.datetime.now()}")
        cloudlog.warning(f"Shutting down manager - {param} set")

    if shutdown:
      break


def main() -> None:
  manager_init()
  if os.getenv("PREPAREONLY") is not None:
    return

  # SystemExit on sigterm
  signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

  try:
    manager_thread()
  except Exception:
    traceback.print_exc()
    sentry.capture_exception()
  finally:
    manager_cleanup()

  params = Params()
  if params.get_bool("DoUninstall"):
    cloudlog.warning("uninstalling")
    HARDWARE.uninstall()
  elif params.get_bool("DoReboot"):
    cloudlog.warning("reboot")
    HARDWARE.reboot()
  elif params.get_bool("DoShutdown"):
    cloudlog.warning("shutdown")
    HARDWARE.shutdown()


if __name__ == "__main__":
  unblock_stdout()

  try:
    main()
  except KeyboardInterrupt:
    print("got CTRL-C, exiting")
  except Exception:
    add_file_handler(cloudlog)
    cloudlog.exception("Manager failed to start")

    try:
      managed_processes['ui'].stop()
    except Exception:
      pass

    # Show last 3 lines of traceback
    error = traceback.format_exc(-3)
    error = "Manager failed to start\n\n" + error
    with TextWindow(error) as t:
      t.wait_for_exit()

    raise

  # manual exit because we are forked
  sys.exit(0)
