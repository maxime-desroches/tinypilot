import copy
from crcmod import mkCrcFun

from common.conversions import MS_TO_KPH
from selfdrive.car.tesla.values import CANBUS, JERK_LIMIT_MIN, JERK_LIMIT_MAX

class TeslaCAN:
    def __init__(self, packer, pt_packer):
        self.packer = packer
        self.pt_packer = pt_packer
        self.crc = mkCrcFun(0x11d, initCrc=0x00, rev=False, xorOut=0xff)

    @staticmethod
    def checksum(msg_id, dat):
        # TODO: get message ID from name instead
        ret = (msg_id & 0xFF) + ((msg_id >> 8) & 0xFF)
        ret += sum(dat)
        return ret & 0xFF
    def create_steering_control(self, angle, enabled, frame):
        values = {
            "DAS_steeringAngleRequest": -angle,
            "DAS_steeringHapticRequest": 0,
            "DAS_steeringControlType": 1 if enabled else 0,
            "DAS_steeringControlCounter": (frame % 16),
        }

        data = self.packer.make_can_msg("DAS_steeringControl", CANBUS.chassis, values)[2]
        values["DAS_steeringControlChecksum"] = self.checksum(0x488, data[:3])
        return self.packer.make_can_msg("DAS_steeringControl", CANBUS.chassis, values)

    def create_action_request(self, msg_stw_actn_req, cancel, bus, counter):
        values = copy.copy(msg_stw_actn_req)

        if cancel is True:
            values["SpdCtrlLvr_Stat"] = 1
            values["MC_STW_ACTN_RQ"] = counter

        data = self.packer.make_can_msg("STW_ACTN_RQ", bus, values)[2]
        values["CRC_STW_ACTN_RQ"] = self.crc(data[:7])
        return self.packer.make_can_msg("STW_ACTN_RQ", bus, values)

    def create_longitudinal_commands(self, acc_state, speed, min_accel, max_accel, cnt):
        messages = []
        values = {
            "DAS_setSpeed": speed * MS_TO_KPH,
            "DAS_accState": acc_state,
            "DAS_aebEvent": 0,
            "DAS_jerkMin": JERK_LIMIT_MIN,
            "DAS_jerkMax": JERK_LIMIT_MAX,
            "DAS_accelMin": min_accel,
            "DAS_accelMax": max_accel,
            "DAS_controlCounter": cnt,
            "DAS_controlChecksum": 0,
        }

        for packer, bus in [self.packer, self.pt_packer], [CANBUS.chassis, CANBUS.powertrain]:
            data = packer.make_can_msg("DAS_control", bus, values)[2]
            values["DAS_controlChecksum"] = self.checksum(0x2b9, data[:7])
            messages.append(packer.make_can_msg("DAS_control", bus, values))
        return messages
