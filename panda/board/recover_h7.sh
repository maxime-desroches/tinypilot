#!/usr/bin/env sh
set -e

DFU_UTIL="dfu-util"

PANDA_H7=1 scons -u

PYTHONPATH=.. python3 -c "from python import Panda; Panda().reset(enter_bootstub=True); Panda().reset(enter_bootloader=True)" || true
sleep 1
$DFU_UTIL -d 0483:df11 -a 0 -s 0x08020000 -D obj/panda_h7.bin.signed
$DFU_UTIL -d 0483:df11 -a 0 -s 0x08000000:leave -D obj/bootstub.panda_h7.bin
