#human steer override module
from selfdrive.config import Conversions as CV
import selfdrive.messaging as messaging
import custom_alert as customAlert
import time

def _current_time_millis():
  return int(round(time.time() * 1000))

class HSOController(object):
    def __init__(self,carcontroller):
        self.CC = carcontroller
        self.frame_humanSteered = 0
    

    def update_stat(self,CS,enabled,actuators,frame):
        human_control = False
        
        if (CS.cstm_btns.get_button_status("steer") >0):
            #if steering but not by ALCA
            if (CS.right_blinker_on or CS.left_blinker_on) and (self.CC.ALCA.laneChange_enabled <= 1):
                self.frame_humanSteered = frame
                customAlert.custom_alert_message("Manual Steering Enabled",CS,50)
            if (CS.steer_override>0): 
                self.frame_humanSteered = frame
                customAlert.custom_alert_message("Manual Steering Enabled",CS,50)
            else:
                if (frame - self.frame_humanSteered < 50): # Need more human testing of handoff timing
                    # Find steering difference between visiond model and human (no need to do every frame if we run out of CPU):
                    steer_current=(CS.angle_steers)  # Formula to convert current steering angle to match apply_steer calculated number
                    apply_steer = int(-actuators.steerAngle)
                    angle = abs(apply_steer-steer_current)
                    # If OP steering > 5 degrees different from human than count that as human still steering..
                    # Tesla rack doesn't report accurate enough, i.e. lane switch we show no human steering when they
                    # still are crossing road at an angle clearly they don't want OP to take over
                    if angle > 50:
                        self.frame_humanSteered = frame
                        customAlert.custom_alert_message("Manual Steering Enabled",CS,50)
            if enabled:
                if (frame - self.frame_humanSteered < 50):
                    human_control = True
                    CS.cstm_btns.set_button_status("steer",3)
                    customAlert.custom_alert_message("Manual Steering Enabled",CS,50)
                else:
                    CS.cstm_btns.set_button_status("steer",2)
            else:
                CS.cstm_btns.set_button_status("steer",1)
        return human_control and enabled

