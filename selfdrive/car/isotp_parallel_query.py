import time
from collections import defaultdict
from functools import partial

import cereal.messaging as messaging
from system.swaglog import cloudlog
from selfdrive.boardd.boardd import can_list_to_can_capnp
from panda.python.uds import CanClient, IsoTpMessage, get_rx_addr_for_tx_addr


class IsoTpParallelQuery:
  def __init__(self, sendcan, logcan, bus, addrs, request, response, response_offset=0x8, functional_addr=False, debug=False, response_pending_timeout=10):
    self.sendcan = sendcan
    self.logcan = logcan
    self.bus = bus
    self.request = request
    self.response = response
    self.debug = debug
    self.response_pending_timeout = response_pending_timeout

    if functional_addr:
      # If functional_addr is True, we query the functional addrs and only process responses from addrs's respective rx addrs
      self.msg_addrs = {}
      for a in addrs:
        functional_tx_addr = 0x7DF if a < 2 ** 11 else 0x18DB33F1
        self.msg_addrs[get_rx_addr_for_tx_addr(a, response_offset)] = (functional_tx_addr, None)

    else:
      real_addrs = [a if isinstance(a, tuple) else (a, None) for a in addrs]
      self.msg_addrs = {get_rx_addr_for_tx_addr(tx_addr[0], rx_offset=response_offset): tx_addr for tx_addr in real_addrs}
    self.msg_buffer = defaultdict(list)

  def rx(self):
    """Drain can socket and sort messages into buffers based on address"""
    can_packets = messaging.drain_sock(self.logcan, wait_for_one=True)

    for packet in can_packets:
      for msg in packet.can:
        if msg.src == self.bus and msg.address in self.msg_addrs.keys():
          self.msg_buffer[msg.address].append((msg.address, msg.busTime, msg.dat, msg.src))

  def _can_tx(self, tx_addr, dat, bus):
    """Helper function to send single message"""
    msg = [tx_addr, 0, dat, bus]
    self.sendcan.send(can_list_to_can_capnp([msg], msgtype='sendcan'))

  def _can_rx(self, addr, sub_addr=None):
    """Helper function to retrieve message with specified address and subadress from buffer"""
    keep_msgs = []

    if sub_addr is None:
      msgs = self.msg_buffer[addr]
    else:
      # Filter based on subadress
      msgs = []
      for m in self.msg_buffer[addr]:
        first_byte = m[2][0]
        if first_byte == sub_addr:
          msgs.append(m)
        else:
          keep_msgs.append(m)

    self.msg_buffer[addr] = keep_msgs
    return msgs

  def _drain_rx(self):
    messaging.drain_sock(self.logcan)
    self.msg_buffer = defaultdict(list)

  def get_data(self, timeout, total_timeout=60.):
    self._drain_rx()

    # Create message objects
    msgs = {}
    request_counter = {}
    request_done = {}
    tx_addrs_sent = []
    for rx_addr, tx_addr in self.msg_addrs.items():
      sub_addr = tx_addr[1]
      max_len = 8 if sub_addr is None else 7

      can_client = CanClient(self._can_tx, partial(self._can_rx, rx_addr, sub_addr=sub_addr), tx_addr[0], rx_addr,
                             self.bus, sub_addr=sub_addr, debug=self.debug)

      msg = IsoTpMessage(can_client, timeout=0, max_len=max_len, debug=self.debug)
      msgs[rx_addr] = msg
      request_counter[rx_addr] = 0
      request_done[rx_addr] = False

      # Send first query to each tx addr once
      if tx_addr not in tx_addrs_sent:
        msg.send(self.request[0])
        tx_addrs_sent.append(tx_addr)

    results = {}
    start_time = time.monotonic()
    response_timeouts = {rx_addr: start_time + timeout for rx_addr in self.msg_addrs}
    while True:
      self.rx()

      if all(request_done.values()):
        break

      for rx_addr, msg in msgs.items():
        try:
          dat, updated = msg.recv()
        except Exception:
          cloudlog.exception(f"Error processing UDS response: {rx_addr}")
          request_done[rx_addr] = True
          continue

        if updated:
          response_timeouts[rx_addr] = time.monotonic() + timeout

        if not dat:
          continue

        counter = request_counter[rx_addr]
        expected_response = self.response[counter]
        response_valid = dat[:len(expected_response)] == expected_response

        if response_valid:
          if counter + 1 < len(self.request):
            msg.send(self.request[counter + 1])
            request_counter[rx_addr] += 1
          else:
            results[(rx_addr, msg._can_client.rx_addr)] = dat[len(expected_response):]
            request_done[rx_addr] = True
        else:
          error_code = dat[2] if len(dat) > 2 else -1
          if error_code == 0x78:
            response_timeouts[rx_addr] = time.monotonic() + self.response_pending_timeout
            if self.debug:
              cloudlog.warning(f"iso-tp query response pending: {rx_addr}")
          else:
            response_timeouts[rx_addr] = 0
            request_done[rx_addr] = True
            cloudlog.warning(f"iso-tp query bad response: {rx_addr} - 0x{dat.hex()}")

      cur_time = time.monotonic()
      if cur_time - max(response_timeouts.values()) > 0:
        for rx_addr in msgs:
          if request_counter[rx_addr] > 0 and not request_done[rx_addr]:
            cloudlog.warning(f"iso-tp query timeout after receiving response: {rx_addr}")
        break

      if cur_time - start_time > total_timeout:
        cloudlog.warning("iso-tp query timeout while receiving data")
        break

    return results
