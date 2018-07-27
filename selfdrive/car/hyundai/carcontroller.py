from common.numpy_fast import clip, interp
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.hyundai.hyundaican import make_can_msg, create_lkas11, create_lkas12b
from selfdrive.car.hyundai.values import ECU, STATIC_MSGS, CAR
from selfdrive.can.packer import CANPacker


# Steer torque limits
STEER_MAX = 200   # Actual limit is about 1023, but not tested, and not needed
STEER_DELTA = 5   # We have no Panda Safety, don't be silly here!   Good idea, YOU add Panda Safety!


TARGET_IDS = [0x340]


class CarController(object):
  def __init__(self, dbc_name, car_fingerprint, enable_camera):
    self.braking = False
    self.controls_allowed = True
    self.last_steer = 0
    self.car_fingerprint = car_fingerprint
    self.angle_control = False
    self.idx = 0
    self.lkas_request = 0
    self.lanes = 0
    self.steer_angle_enabled = False
    self.ipas_reset_counter = 0
    self.turning_inhibit = 0
    self.limited_steer = 0
    self.hide_lkas_fault = 100
    print self.car_fingerprint

    self.fake_ecus = set()
    if enable_camera: self.fake_ecus.add(ECU.CAM)
    self.packer = CANPacker(dbc_name)

  def update(self, sendcan, enabled, CS, frame, actuators, CamS):
    # When the driver starts inputting torque, reduce the requested torque
    #  We do not want to remove torque because we want to know if OP is still steering
    #  And how we do it, meh, who cares, this might do.
    if abs(CS.steer_torque_driver) > 1:
      self.limited_steer = actuators.steer / (abs(CS.steer_torque_driver) * 2)
    else:
      self.limited_steer = actuators.steer

    # Steering Torque Scaling is to STEER_MAX, + 1024 for center
    apply_steer = int(round((self.limited_steer * STEER_MAX) + 1024))

    # This is redundant clipping code, kept in case it needs to be advanced
    max_lim = 1024 + STEER_MAX
    min_lim = 1024 - STEER_MAX
    apply_steer = clip(apply_steer, min_lim, max_lim)

    # Very basic Rate Limiting
    # TODO: Revisit this
    if (apply_steer - self.last_steer) > STEER_DELTA:
      apply_steer = self.last_steer + STEER_DELTA
    elif (self.last_steer - apply_steer) > STEER_DELTA:
      apply_steer = self.last_steer - STEER_DELTA


    # Inhibits *outside of* alerts
    #    Because the Turning Indicator Status is based on Lights and not Stalk, latching is 
    #    needed for the disable to work.
    if CS.left_blinker_on == 1 or CS.right_blinker_on == 1:
      self.turning_inhibit = 180  # Disable for 1.8 Seconds after blinker turned off

    if self.turning_inhibit > 0:
      self.turning_inhibit = self.turning_inhibit - 1

    if not enabled or self.turning_inhibit > 0:
      apply_steer = 1024     # 1024 is midpoint (no steer)
      self.last_steer = 1024 # Reset Last Steer
    else:
      self.lanes = 3 * 4     # bit 0 = Right Lane, bit 1 = Left Lane, Offset by 2 bits in byte.

    self.last_steer = apply_steer

    can_sends = []

    # Index is 4 bits long, this is the counter
    self.idx = self.idx + 1
    if self.idx >= 16:
      self.idx = 0
    
    # Byte 4 is used for Index and HBA
    #   We generate the Index, but pass through HBA
    lkas11_byte4 = self.idx * 16

    # Split apply steer Word into 2 Bytes
    apply_steer_a = apply_steer & 0xFF
    apply_steer_b = (apply_steer >> 8) & 0xFF


    # If Request to Steer is anything but 0 torque, turn on ActToi
    if apply_steer != 1024:
      apply_steer_b = apply_steer_b + 0x08


    if enabled:
      # When we send Torque signals that the camera does not expet, it faults.
      #   This masks the fault for 750ms after bringing stock back on.
      #   This does NOT mean that the factory system will be enabled, it will still be off.
      #      This was tested at 500ms, and 1 in 10 disables, a fault was still seen.
      self.hide_lkas_fault = 75

      # Generate the 7 bytes as needed for OP Control.
      #   Anything we don't generate, pass through from camera
      lkas11_byte0 = int(self.lanes) + (CamS.lkas11_b0 & 0xC3)
      lkas11_byte1 = CamS.lkas11_b1
      lkas11_byte2 = apply_steer_a
      lkas11_byte3 = apply_steer_b + (CamS.lkas11_b3 & 0xE0)   # ToiFlt always comes on, don't pass it
      lkas11_byte4 = lkas11_byte4 + (CamS.lkas11_b4 & 0x0F)    # Always use our counter
      lkas11_byte5 = CamS.lkas11_b5
      lkas11_byte7 = CamS.lkas11_b7
    else:
      # Pass Through the 7 bytes so that Factory LKAS is in control
      #   We still use our counter, because otherwise duplicates and missed messages from the camera
      #   is possible due to the implementation method.  As such, we recreate the checksum as well
      # Byte 0 defined below due to Fault Masking
      lkas11_byte1 = CamS.lkas11_b1
      lkas11_byte2 = CamS.lkas11_b2
      # Byte 3 defined below due to Fault Masking
      lkas11_byte4 = lkas11_byte4 + (CamS.lkas11_b4 & 0x0F)    # Always use our counter
      lkas11_byte5 = CamS.lkas11_b5
      lkas11_byte7 = CamS.lkas11_b7
      # This is the Fault Masking needed in byte 0 and byte 3
      if self.hide_lkas_fault > 0:
        lkas11_byte0 = int(self.lanes) + (CamS.lkas11_b0 & 0xC3)
        lkas11_byte3 = CamS.lkas11_b3 & 0xE7
        self.hide_lkas_fault = self.hide_lkas_fault - 1
      else:
        lkas11_byte0 = CamS.lkas11_b0
        lkas11_byte3 = CamS.lkas11_b3
        


    # Create Checksum
    #   Sorento checksum is Byte 0 to 5
    #   Other models appear to be Byte 0 to Byte 5 as well as Byte 7
    if CS.car_fingerprint == CAR.SORENTO:
      # 6 Byte Checksum
      checksum = (lkas11_byte0 + lkas11_byte1 + lkas11_byte2 + lkas11_byte3 + \
        lkas11_byte4 + lkas11_byte5) % 256
    else:
      # 7 Byte Checksum
      checksum = (lkas11_byte0 + lkas11_byte1 + lkas11_byte2 + lkas11_byte3 + \
        lkas11_byte4 + lkas11_byte5 + lkas11_byte7) % 256
    


    # Create LKAS11 Message at 100Hz
    can_sends.append(create_lkas11(self.packer, lkas11_byte0, \
      lkas11_byte1, lkas11_byte2, lkas11_byte3, lkas11_byte4, \
      lkas11_byte5, checksum, lkas11_byte7))

   

    # Create LKAS12 Message at 10Hz
    if (frame % 10) == 0:
      can_sends.append(create_lkas12b(self.packer, CamS.lkas12_b0, CamS.lkas12_b1, \
        CamS.lkas12_b2, CamS.lkas12_b3, CamS.lkas12_b4, CamS.lkas12_b5))



    # Limit Terminal Debugging to 5Hz
    if (frame % 20) == 0:
      print "controlsdDebug steer", actuators.steer, "strToq", CS.steer_torque_driver, "lim steer", self.limited_steer, "v_ego", \
        CS.v_ego, "strAng", CS.angle_steers


    
    # Send messages to canbus
    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
