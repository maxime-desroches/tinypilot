#!/usr/bin/env python3
import paramiko  # pylint: disable=import-error
import os
import sys
import re
import time
import socket


SOURCE_DIR = "/data/openpilot_source"
TEST_DIR = "/data/openpilot"

def run_on_phone(test_cmd):

  # connect to phone over SSH
  eon_ip = os.environ.get('eon_ip', None)
  if eon_ip is None:
    raise Exception("'eon_ip' not set")

  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  key_file = open(os.path.join(os.path.dirname(__file__), "../../tools/ssh/key/id_rsa"))
  key = paramiko.RSAKey.from_private_key(key_file)

  print("SSH to phone {}".format(eon_ip))

  # Try connecting for one minute
  t_start = time.time()
  while True:
    try:
      ssh.connect(hostname=eon_ip, port=8022, pkey=key, timeout=10)
    except (paramiko.ssh_exception.SSHException, socket.timeout, paramiko.ssh_exception.NoValidConnectionsError):
      print("Connection failed")
      if time.time() - t_start > 60:
        raise
    else:
      break
    time.sleep(1)

  # set up environment
  conn = ssh.invoke_shell()
  branch = os.environ['GIT_BRANCH']
  commit = os.environ.get('GIT_COMMIT', branch)

  conn.send(f"cd {SOURCE_DIR}\n")
  conn.send("git reset --hard\n")
  conn.send("git fetch origin\n")
  conn.send("git checkout %s\n" % commit)
  conn.send("git clean -xdf\n")
  conn.send("git submodule update --init\n")
  conn.send("git submodule foreach --recursive git reset --hard\n")
  conn.send("git submodule foreach --recursive git clean -xdf\n")
  conn.send("echo \"git took $SECONDS seconds\"\n")

  conn.send(f"rm -rf {TEST_DIR}\n")
  conn.send(f"cp -R {SOURCE_DIR} {TEST_DIR}\n")

  # run the test
  conn.send(test_cmd + "\n")

  # get the result and print it back out
  conn.send('echo "RESULT:" $?\n')
  conn.send("exit\n")

  dat = b""

  while True:
    recvd = conn.recv(4096)
    if len(recvd) == 0:
      break

    dat += recvd
    sys.stdout.buffer.write(recvd)
    sys.stdout.flush()

  returns = re.findall(rb'^RESULT: (\d+)', dat[-1024:], flags=re.MULTILINE)
  sys.exit(int(returns[0]))


if __name__ == "__main__":
  run_on_phone(sys.argv[1])
