#!/usr/bin/env python3
import datetime
import os
import sys
import signal
import subprocess
import traceback

from common.basedir import BASEDIR
from common.spinner import Spinner
from common.text_window import TextWindow
import selfdrive.crash as crash
from selfdrive.manager.helpers import build, unblock_stdout, MAX_BUILD_PROGRESS
from selfdrive.hardware import HARDWARE, EON, PC, TICI
from selfdrive.hardware.eon.apk import update_apks, pm_apply_packages, start_offroad
from selfdrive.swaglog import cloudlog, add_logentries_handler
from selfdrive.version import version, dirty


os.environ['BASEDIR'] = BASEDIR
sys.path.append(os.path.join(BASEDIR, "pyextra"))

WEBCAM = os.getenv("WEBCAM") is not None
PREBUILT = os.path.exists(os.path.join(BASEDIR, 'prebuilt'))


if __name__ == "__main__":
  unblock_stdout()


# Start spinner
spinner = Spinner()
spinner.update_progress(0, 100)
if __name__ != "__main__":
  spinner.close()


if __name__ == "__main__" and not PREBUILT:
  build(spinner, dirty)

import cereal.messaging as messaging
from common.params import Params
from selfdrive.registration import register
from selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess, manage_processes


managed_processes = [
  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),
  # due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
  NativeProcess("camerad", "selfdrive/camerad", ["./camerad"], unkillable=True, driverview=True),
  NativeProcess("clocksd", "selfdrive/clocksd", ["./clocksd"]),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], driverview=True),
  NativeProcess("logcatd", "selfdrive/logcatd", ["./logcatd"]),
  NativeProcess("loggerd", "selfdrive/loggerd", ["./loggerd"]),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"]),
  NativeProcess("proclogd", "selfdrive/proclogd", ["./proclogd"]),
  NativeProcess("sensord", "selfdrive/sensord", ["./sensord"], persistent=EON, sigkill=EON),
  NativeProcess("ubloxd", "selfdrive/locationd", ["./ubloxd"]),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], persistent=True),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd"),
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "selfdrive.loggerd.deleter", persistent=True),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", driverview=True),
  PythonProcess("locationd", "selfdrive.locationd.locationd"),
  PythonProcess("logmessaged", "selfdrive.logmessaged", persistent=True),
  PythonProcess("pandad", "selfdrive.pandad", persistent=True),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("thermald", "selfdrive.thermald.thermald", persistent=True),
  PythonProcess("uploader", "selfdrive.loggerd.uploader", persistent=True),
]

if not PC:
  managed_processes += [
    PythonProcess("tombstoned", "selfdrive.tombstoned", persistent=True),
    PythonProcess("updated", "selfdrive.updated", persistent=True),
  ]

if TICI:
  managed_processes += [
    PythonProcess("timezoned", "selfdrive.timezoned", persistent=True),
  ]

if EON:
  managed_processes += [
    PythonProcess("rtshield", "selfdrive.rtshield"),
  ]


def cleanup_all_processes(signal, frame):
  cloudlog.info("caught ctrl-c %s %s" % (signal, frame))

  if EON:
    pm_apply_packages('disable')

  for p in managed_processes:
    p.stop()

  cloudlog.info("everything is dead")


# ****************** run loop ******************

def manager_init():
  os.umask(0)  # Make sure we can create files with 777 permissions

  # Create folders needed for msgq
  try:
    os.mkdir("/dev/shm")
  except FileExistsError:
    pass
  except PermissionError:
    print("WARNING: failed to make /dev/shm")

  # set dongle id
  reg_res = register(spinner)
  if reg_res:
    dongle_id = reg_res
  else:
    raise Exception("server registration failed")
  os.environ['DONGLE_ID'] = dongle_id

  if not dirty:
    os.environ['CLEAN'] = '1'

  cloudlog.bind_global(dongle_id=dongle_id, version=version, dirty=dirty,
                       device=HARDWARE.get_device_type())
  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty, device=HARDWARE.get_device_type())

  # ensure shared libraries are readable by apks
  if EON:
    os.chmod(BASEDIR, 0o755)
    os.chmod("/dev/shm", 0o777)
    os.chmod(os.path.join(BASEDIR, "cereal"), 0o755)
    os.chmod(os.path.join(BASEDIR, "cereal", "libmessaging_shared.so"), 0o755)


def manager_thread():
  cloudlog.info("manager start")
  cloudlog.info({"environ": os.environ})

  # save boot log
  subprocess.call("./bootlog", cwd=os.path.join(BASEDIR, "selfdrive/loggerd"))

  ignore = []
  if os.getenv("NOBOARD") is not None:
    ignore.append("pandad")
  if os.getenv("BLOCK") is not None:
    ignore += os.getenv("BLOCK").split(",")

  # start offroad
  if EON and "QT" not in os.environ:
    pm_apply_packages('enable')
    start_offroad()

  started_prev = False
  params = Params()
  sm = messaging.SubMaster(['deviceState'])
  pm = messaging.PubMaster(['managerState'])

  while True:
    sm.update()
    not_run = ignore[:]

    if sm['deviceState'].freeSpacePercent < 5:
      not_run.append("loggerd")

    started = sm['deviceState'].started
    driverview = params.get("IsDriverViewEnabled") == b"1"
    manage_processes(managed_processes, started, driverview, not_run)

    # trigger an update after going offroad
    if started_prev and not started:
      os.sync()
      # TODO
      # send_managed_process_signal("updated", signal.SIGHUP)
    started_prev = started

    running_list = ["%s%s\u001b[0m" % ("\u001b[32m" if p.proc.is_alive() else "\u001b[31m", p.name)
                    for p in managed_processes if p.proc]
    cloudlog.debug(' '.join(running_list))

    # send managerState
    msg = messaging.new_message('managerState')
    msg.managerState.processes = [p.get_process_state_msg() for p in managed_processes]
    pm.send('managerState', msg)

    # Exit main loop when uninstall is needed
    if params.get("DoUninstall", encoding='utf8') == "1":
      break


def manager_prepare():
  # build all processes
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  total = 100.0 - (0 if PREBUILT else MAX_BUILD_PROGRESS)

  for i, p in enumerate(managed_processes):
    perc = (100.0 - total) + total * (i + 1) / len(managed_processes)
    spinner.update_progress(perc, 100.)
    p.prepare()


def main():
  params = Params()
  params.manager_start()

  default_params = [
    ("CommunityFeaturesToggle", "0"),
    ("CompletedTrainingVersion", "0"),
    ("IsRHD", "0"),
    ("IsMetric", "0"),
    ("RecordFront", "0"),
    ("HasAcceptedTerms", "0"),
    ("HasCompletedSetup", "0"),
    ("IsUploadRawEnabled", "1"),
    ("IsLdwEnabled", "1"),
    ("LastUpdateTime", datetime.datetime.utcnow().isoformat().encode('utf8')),
    ("OpenpilotEnabledToggle", "1"),
    ("VisionRadarToggle", "0"),
    ("LaneChangeEnabled", "1"),
    ("IsDriverViewEnabled", "0"),
  ]

  # set unset params
  for k, v in default_params:
    if params.get(k) is None:
      params.put(k, v)

  # is this dashcam?
  if os.getenv("PASSIVE") is not None:
    params.put("Passive", str(int(os.getenv("PASSIVE"))))

  if params.get("Passive") is None:
    raise Exception("Passive must be set to continue")

  if EON:
    update_apks()
  manager_init()
  manager_prepare()
  spinner.close()

  if os.getenv("PREPAREONLY") is not None:
    return

  # SystemExit on sigterm
  signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

  try:
    manager_thread()
  except Exception:
    traceback.print_exc()
    crash.capture_exception()
  finally:
    cleanup_all_processes(None, None)

  if params.get("DoUninstall", encoding='utf8') == "1":
    cloudlog.warning("uninstalling")
    HARDWARE.uninstall()


if __name__ == "__main__":
  try:
    main()
  except Exception:
    add_logentries_handler(cloudlog)
    cloudlog.exception("Manager failed to start")

    # Show last 3 lines of traceback
    error = traceback.format_exc(-3)
    error = "Manager failed to start\n\n" + error
    spinner.close()
    with TextWindow(error) as t:
      t.wait_for_exit()

    raise

  # manual exit because we are forked
  sys.exit(0)
