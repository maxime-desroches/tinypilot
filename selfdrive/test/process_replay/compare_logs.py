#!/usr/bin/env python3
import sys
import math
import capnp
import numbers
import dictdiffer
from collections import Counter, defaultdict

from openpilot.tools.lib.logreader import LogReader

EPSILON = sys.float_info.epsilon


def remove_ignored_fields(msg, ignore):
  msg = msg.as_builder()
  for key in ignore:
    attr = msg
    keys = key.split(".")
    if msg.which() != keys[0] and len(keys) > 1:
      continue

    for k in keys[:-1]:
      # indexing into list
      if k.isdigit():
        attr = attr[int(k)]
      else:
        attr = getattr(attr, k)

    v = getattr(attr, keys[-1])
    if isinstance(v, bool):
      val = False
    elif isinstance(v, numbers.Number):
      val = 0
    elif isinstance(v, (list, capnp.lib.capnp._DynamicListBuilder)):
      val = []
    else:
      raise NotImplementedError(f"Unknown type: {type(v)}")
    setattr(attr, keys[-1], val)
  return msg


class CapnpSet(set):
  pass

class HashableCapnpStructBuilder:
  def __init__(self, msg):
    self.msg = msg


def compare_logs(log1, log2, ignore_fields=None, ignore_msgs=None, tolerance=None,):
  if ignore_fields is None:
    ignore_fields = []
  if ignore_msgs is None:
    ignore_msgs = []
  tolerance = EPSILON if tolerance is None else tolerance

  log1, log2 = (
    [m for m in log if m.which() not in ignore_msgs]
    for log in (log1, log2)
  )

  msgs_by_which_log1 = defaultdict(list)
  msgs_by_which_log2 = defaultdict(list)

  for msg1 in log1:
    # if msg1.which() == 'carEvents':
    #   continue
    msgs_by_which_log1[msg1.which()].append(msg1)
  for msg2 in log2:
    msgs_by_which_log2[msg2.which()].append(msg2)

  if set(msgs_by_which_log1) != set(msgs_by_which_log2):
    raise Exception(f"logs service keys don't match:\n\t\t{set(msgs_by_which_log1)}\n\t\t{set(msgs_by_which_log2)}")

  diff = []
  print('IGNORE', ignore_fields)
  for which in msgs_by_which_log1.keys():
    print(which, len(msgs_by_which_log1[which]), len(msgs_by_which_log2[which]))
    if len(msgs_by_which_log1[which]) == len(msgs_by_which_log2[which]):
      print(which, 'log length equal')
      for msg1, msg2 in zip(msgs_by_which_log1[which], msgs_by_which_log2[which], strict=True):
        # if msg1.which() != msg2.which():
        #   raise Exception("msgs not aligned between logs")
        # continue

        msg1 = remove_ignored_fields(msg1, ignore_fields)
        msg2 = remove_ignored_fields(msg2, ignore_fields)

        if msg1.to_bytes() != msg2.to_bytes():
          msg1_dict = msg1.as_reader().to_dict(verbose=True)
          msg2_dict = msg2.as_reader().to_dict(verbose=True)

          dd = dictdiffer.diff(msg1_dict, msg2_dict, ignore=ignore_fields)

          # Dictdiffer only supports relative tolerance, we also want to check for absolute
          # TODO: add this to dictdiffer
          def outside_tolerance(diff):
            try:
              if diff[0] == "change":
                a, b = diff[2]
                finite = math.isfinite(a) and math.isfinite(b)
                if finite and isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
                  return abs(a - b) > max(tolerance, tolerance * max(abs(a), abs(b)))
            except TypeError:
              pass
            return True

          dd = list(filter(outside_tolerance, dd))

          diff.extend(dd)
    else:
      print(which, 'print diff in logs')
      new_msgs1 = [remove_ignored_fields(msg1, ignore_fields).as_reader().to_dict(verbose=True) for msg1 in msgs_by_which_log1[which]]
      new_msgs2 = [remove_ignored_fields(msg2, ignore_fields).as_reader().to_dict(verbose=True) for msg2 in msgs_by_which_log2[which]]
      added_msgs = [m2 for m2 in new_msgs2 if m2 not in new_msgs1]
      removed_msgs = [m1 for m1 in new_msgs1 if m1 not in new_msgs2]
      print('--- START ---')
      for msg in added_msgs:
        print('ADDED MSG', msg)
      print('---')
      for msg in removed_msgs:
        print('REMOVED MSG', msg)
      print('--- end ---')
      dd_add = []
      dd = []
      dd.extend([list(*dictdiffer.diff(r, {}, ignore=ignore_fields)) for r in removed_msgs])
      dd.extend([list(*dictdiffer.diff({}, a, ignore=ignore_fields)) for a in added_msgs])
      # diff.extend([('removed', which, list(dictdiffer.diff(r, {}, ignore=ignore_fields))[0][2]) for r in removed_msgs])
      # diff.extend([('added', which, list(dictdiffer.diff({}, a, ignore=ignore_fields))[0][2]) for a in added_msgs])
      dd = [(typ, which, dif) for typ, _, dif in dd]
      diff.extend(dd)
      # diff.extend([list(*dictdiffer.diff(r, {}, ignore=ignore_fields)) for r in removed_msgs])
      # diff.extend([list(*dictdiffer.diff({}, a, ignore=ignore_fields)) for a in added_msgs])
      print('diff', diff)
      break
      # dd_add += [list(dictdiffer.diff([], a, ignore=ignore_fields)) for a in added_msgs]
      # print('dd_add')
      # print(dd_add)
      # dd_add = list(dictdiffer.diff({}, added_msgs, ignore=ignore_fields))
      # dd_add = [list(dictdiffer.diff(new_msgs1, new_msgs2, ignore=ignore_fields))]
      # diff.append('Added messages:')
      # diff.extend(map(str, added_msgs))
      # diff.append('')
      # diff.append('Added/removed messages:')
      # diff.extend(map(str, removed_msgs))
      # diff.extend(dd_add)
      print('Added msgs:', len(added_msgs))
      print('Removed msgs:', len(removed_msgs))
      continue
      print("Added msgs: {}".format("\n".join(added_msgs)))
      print("Removed msgs: {}".format("\n".join(removed_msgs)))
      # added_msgs = set(new_msgs2) - set(new_msgs1)
      # removed_msgs = set(new_msgs1) - set(new_msgs2)
      print('Added msgs:', len(added_msgs))
      print('Removed msgs:', len(removed_msgs))

  return diff


if __name__ == "__main__":
  log1 = list(LogReader(sys.argv[1]))
  log2 = list(LogReader(sys.argv[2]))
  print(compare_logs(log1, log2, sys.argv[3:]))
