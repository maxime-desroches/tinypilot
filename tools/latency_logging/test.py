import cereal.messaging as messaging
import capnp

# in subscriber
sm = messaging.SubMaster(["lateralPlan", "sendcan"])
while 1:
  sm.update()
  for a in sm.updated:
      print(sm[a])
