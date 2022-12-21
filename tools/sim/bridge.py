#!/usr/bin/env python3
import argparse
import math
import os
import signal
import threading
import time
from multiprocessing import Process, Queue
from typing import Any

import carla  # pylint: disable=import-error
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

import cereal.messaging as messaging
from cereal import log
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.basedir import BASEDIR
from common.params import Params
from common.realtime import DT_DMON, Ratekeeper
from selfdrive.car.honda.values import CruiseButtons
from selfdrive.test.helpers import set_params_enabled
from tools.sim.lib.can import can_function

W, H = 1928, 1208
PRINT_DECIMATION = 100

pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'accelerometer', 'gyroscope', 'can', "gpsLocationExternal"])
sm = messaging.SubMaster(['carControl', 'controlsState'])

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  parser.add_argument('--town', type=str, default='Town04_Opt')
  parser.add_argument('--spawn_point', dest='num_selected_spawn_point', type=int, default=16)
  parser.add_argument('--host', dest='host', type=str, default='127.0.0.1')
  parser.add_argument('--port', dest='port', type=int, default=2000)

  return parser.parse_args(add_args)


class VehicleState:
  def __init__(self):
    self.speed = 0.0
    self.angle = 0.0
    self.bearing_deg = 0.0
    self.vel = carla.Vector3D()
    self.cruise_button = 0
    self.is_engaged = False
    self.ignition = True


class TrottleBrakeSteer:
  def __init__(self, throttle=0, brake=0, steer=0):
    self.reset(throttle=0, brake=0, steer=0)

  def reset(self, throttle=0, brake=0, steer=0):
    self.throttle = throttle
    self.brake = brake
    self.steer = steer

  def __repr__(self):
    return "[T:%.4f B:%.4f S:%.4f]" % (self.throttle, self.brake, self.steer)

  def __eq__(self, other):
    return (self.throttle, self.brake, self.steer) == (other.throttle, other.brake, other.steer)


def clamp(num, bound2, bound1):
  if bound2 < bound1:
    return max(min(num, bound1), bound2)
  else:
    return max(min(num, bound2), bound1)


def normalize(values, actual_bounds, desired_bounds):
  return desired_bounds[0] + (clamp(values, actual_bounds[0], actual_bounds[1]) - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0])


def rate_limit(old, new, limit):
  if new > old + limit:
    result = old + limit
  elif new < old - limit:
    result = old - limit
  else:
    result = new
  return result


def TBS_scale_clamp(tbs, scalingtype):
  if scalingtype == 'openpilot2carla':
    tbs.throttle = normalize(tbs.throttle, (0, 1), (0, 1))
    tbs.brake = normalize(tbs.brake, (0, 1), (0, 0.25))
    # tbs.steer / (-1000) # normalize(tbs.steer, (-100, 100), (0.1, -0.1) )      <- Exceed OP steering limit
    tbs.steer = normalize(tbs.steer, (-100, 100), (0.1, -0.1))
  else:  # manual2carla
    tbs.throttle = normalize(tbs.throttle, (0, 1), (0, 1))
    tbs.brake = normalize(tbs.brake, (0, 1), (0, 0.7))
    tbs.steer = normalize(tbs.steer, (-1, 1), (1, -1))
  return tbs  # no scaling


def TBS_rate_limit(old, new, mode):
  if mode == 'openpilot':
    Tlimit = 0.001
    Blimit = 1
    Slimit = 0.0002
  else:  # manual
    # Make manual losing throttle gradually
    if new.throttle == 0:
      Tlimit = 0.001
    else:
      Tlimit = 1
    if new.brake == 0:
      Blimit = 0.1
    else:
      Blimit = 1
    Slimit = 1
  return TrottleBrakeSteer(throttle=rate_limit(old.throttle, new.throttle, Tlimit), brake=rate_limit(old.brake, new.brake, Blimit), steer=rate_limit(old.steer, new.steer, Slimit))


class Camerad:
  def __init__(self):
    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, W, H)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, W, H)
    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)
    cl_arg = f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W * 3} -DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG "

    kernel_fn = os.path.join(BASEDIR, "tools/sim/rgb_to_nv12.cl")
    with open(kernel_fn) as f:
      prg = cl.Program(self.ctx, f.read()).build(cl_arg)
      self.krnl = prg.rgb_to_nv12
    self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
    self.Hdiv4 = H // 4 if (H % 4 == 0) else (H + (4 - H % 4)) // 4

  def cam_callback_road(self, image):
    self._cam_callback(image, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_callback_wide_road(self, image):
    self._cam_callback(image, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def _cam_callback(self, image, frame_id, pub_type, yuv_type):
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (H, W, 4))
    img = img[:, :, [0, 1, 2]].copy()

    # convert RGB frame to YUV
    rgb = np.reshape(img, (H, W * 3))
    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)
    self.krnl(self.queue, (np.int32(self.Wdiv4), np.int32(self.Hdiv4)), None, rgb_cl.data, yuv_cl.data).wait()
    yuv = np.resize(yuv_cl.get(), rgb.size // 2)
    eof = int(frame_id * 0.05 * 1e9)

    self.vipc_server.send(yuv_type, yuv.data.tobytes(), frame_id, eof, eof)

    dat = messaging.new_message(pub_type)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    pm.send(pub_type, dat)

def imu_callback(imu, vehicle_state):
  # send 5x since 'sensor_tick' doesn't seem to work. limited by the world tick?
  for _ in range(5):
    vehicle_state.bearing_deg = math.degrees(imu.compass)
    dat = messaging.new_message('accelerometer')
    dat.accelerometer.sensor = 4
    dat.accelerometer.type = 0x10
    dat.accelerometer.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.accelerometer.init('acceleration')
    dat.accelerometer.acceleration.v = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]
    pm.send('accelerometer', dat)

    # copied these numbers from locationd
    dat = messaging.new_message('gyroscope')
    dat.gyroscope.sensor = 5
    dat.gyroscope.type = 0x10
    dat.gyroscope.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.gyroscope.init('gyroUncalibrated')
    dat.gyroscope.gyroUncalibrated.v = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
    pm.send('gyroscope', dat)
    time.sleep(0.01)


def panda_state_function(vs: VehicleState, exit_event: threading.Event):
  pm = messaging.PubMaster(['pandaStates'])
  while not exit_event.is_set():
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': vs.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec'
    }
    pm.send('pandaStates', dat)
    time.sleep(0.5)


def peripheral_state_function(exit_event: threading.Event):
  pm = messaging.PubMaster(['peripheralState'])
  while not exit_event.is_set():
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    # fake peripheral state data
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    pm.send('peripheralState', dat)
    time.sleep(0.5)


def gps_callback(gps, vehicle_state):
  dat = messaging.new_message('gpsLocationExternal')

  # transform vel from carla to NED
  # north is -Y in CARLA
  velNED = [
    -vehicle_state.vel.y,  # north/south component of NED is negative when moving south
    vehicle_state.vel.x,  # positive when moving east, which is x in carla
    vehicle_state.vel.z,
  ]

  dat.gpsLocationExternal = {
    "unixTimestampMillis": int(time.time() * 1000),
    "flags": 1,  # valid fix
    "accuracy": 1.0,
    "verticalAccuracy": 1.0,
    "speedAccuracy": 0.1,
    "bearingAccuracyDeg": 0.1,
    "vNED": velNED,
    "bearingDeg": vehicle_state.bearing_deg,
    "latitude": gps.latitude,
    "longitude": gps.longitude,
    "altitude": gps.altitude,
    "speed": vehicle_state.speed,
    "source": log.GpsLocationData.SensorSource.ublox,
  }

  pm.send('gpsLocationExternal', dat)


def fake_driver_monitoring(exit_event: threading.Event):
  pm = messaging.PubMaster(['driverStateV2', 'driverMonitoringState'])
  while not exit_event.is_set():
    # dmonitoringmodeld output
    dat = messaging.new_message('driverStateV2')
    dat.driverStateV2.leftDriverData.faceProb = 1.0
    pm.send('driverStateV2', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState')
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    pm.send('driverMonitoringState', dat)

    time.sleep(DT_DMON)


def can_function_runner(vs: VehicleState, exit_event: threading.Event):
  i = 0
  while not exit_event.is_set():
    can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged)
    time.sleep(0.01)
    i += 1


def connect_carla_client(host: str, port: int):
  client = carla.Client(host, port)
  client.set_timeout(5)
  return client


class CarlaBridge:

  def __init__(self, arguments):
    set_params_enabled()

    self.params = Params()

    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = 20
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    self.params.put("CalibrationParams", msg.to_bytes())
    self.params.put_bool("WideCameraOnly", not arguments.dual_camera)

    self._args = arguments
    self._carla_objects = []
    self._camerad = None
    self._exit_event = threading.Event()
    self._threads = []
    self._keep_alive = True
    self.started = False
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self._exit = threading.Event()

  def _on_shutdown(self, signal, frame):
    self._keep_alive = False

  def bridge_keep_alive(self, q: Queue, retries: int):
    try:
      while self._keep_alive:
        try:
          self._run(q)
          break
        except RuntimeError as e:
          self.close()
          if retries == 0:
            raise

          # Reset for another try
          self._carla_objects = []
          self._threads = []
          self._exit_event = threading.Event()

          retries -= 1
          if retries <= -1:
            print(f"Restarting bridge. Error: {e} ")
          else:
            print(f"Restarting bridge. Retries left {retries}. Error: {e} ")
    finally:
      # Clean up resources in the opposite order they were created.
      self.close()

  def _run(self, q: Queue):
    client = connect_carla_client(self._args.host, self._args.port)
    world = client.load_world(self._args.town)

    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearSunset)

    if not self._args.high_quality:
      world.unload_map_layer(carla.MapLayer.Foliage)
      world.unload_map_layer(carla.MapLayer.Buildings)
      world.unload_map_layer(carla.MapLayer.ParkedVehicles)
      world.unload_map_layer(carla.MapLayer.Props)
      world.unload_map_layer(carla.MapLayer.StreetLights)
      world.unload_map_layer(carla.MapLayer.Particles)

    blueprint_library = world.get_blueprint_library()

    world_map = world.get_map()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
    vehicle_bp.set_attribute('role_name', 'hero')
    spawn_points = world_map.get_spawn_points()
    assert len(spawn_points) > self._args.num_selected_spawn_point, f'''No spawn point {self._args.num_selected_spawn_point}, try a value between 0 and
      {len(spawn_points)} for this town.'''
    spawn_point = spawn_points[self._args.num_selected_spawn_point]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    self._carla_objects.append(vehicle)
    # max_steer_angle = vehicle.get_physics_control().wheels[0].max_steer_angle

    # make tires less slippery
    # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
    physics_control = vehicle.get_physics_control()
    physics_control.mass = 2326
    # physics_control.wheels = [wheel_control]*4
    physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
    physics_control.gear_switch_time = 0.0
    vehicle.apply_physics_control(physics_control)

    transform = carla.Transform(carla.Location(x=0.8, z=1.13))

    def create_camera(fov, callback):
      blueprint = blueprint_library.find('sensor.camera.rgb')
      blueprint.set_attribute('image_size_x', str(W))
      blueprint.set_attribute('image_size_y', str(H))
      blueprint.set_attribute('fov', str(fov))
      if not self._args.high_quality:
        blueprint.set_attribute('enable_postprocess_effects', 'False')
      camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
      camera.listen(callback)
      return camera

    self._camerad = Camerad()

    if self._args.dual_camera:
      road_camera = create_camera(fov=40, callback=self._camerad.cam_callback_road)
      self._carla_objects.append(road_camera)

    road_wide_camera = create_camera(fov=120, callback=self._camerad.cam_callback_wide_road)  # fov bigger than 120 shows unwanted artifacts
    self._carla_objects.append(road_wide_camera)

    vehicle_state = VehicleState()

    # re-enable IMU
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.01')
    imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    imu.listen(lambda imu: imu_callback(imu, vehicle_state))

    gps_bp = blueprint_library.find('sensor.other.gnss')
    gps = world.spawn_actor(gps_bp, transform, attach_to=vehicle)
    gps.listen(lambda gps: gps_callback(gps, vehicle_state))
    self.params.put_bool("UbloxAvailable", True)

    self._carla_objects.extend([imu, gps])
    # launch fake car threads
    self._threads.append(threading.Thread(target=panda_state_function, args=(vehicle_state, self._exit_event,)))
    self._threads.append(threading.Thread(target=peripheral_state_function, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=fake_driver_monitoring, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=can_function_runner, args=(vehicle_state, self._exit_event,)))
    for t in self._threads:
      t.start()


    vc = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)

    is_openpilot_engaged = False

    # input
    op= TrottleBrakeSteer()
    manual= TrottleBrakeSteer()
    # result
    out= TrottleBrakeSteer()
    # keeping previous state for change rate limit
    old = TrottleBrakeSteer()

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      world.tick()

    # loop
    rk = Ratekeeper(100, print_delay_threshold=0.05)

    while self._keep_alive:
      # 1. Read the throttle, steer and brake from op or manual controls
      # 2. Set instructions in Carla
      # 3. Send current carstate to op via can

      cruise_button = 0
      manual.reset()
      # --------------Step 1-------------------------------
      if not q.empty():
        message = q.get()
        m = message.split('_')
        if m[0] == "steer":
          manual.steer = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "throttle":
          manual.throttle = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "brake":
          manual.brake = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "reverse":
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False
        elif m[0] == "cruise":
          if m[1] == "down":
            cruise_button = CruiseButtons.DECEL_SET
            is_openpilot_engaged = True
          elif m[1] == "up":
            cruise_button = CruiseButtons.RES_ACCEL
            is_openpilot_engaged = True
          elif m[1] == "cancel":
            cruise_button = CruiseButtons.CANCEL
            is_openpilot_engaged = False
        elif m[0] == "ignition":
          vehicle_state.ignition = not vehicle_state.ignition
        elif m[0] == "quit":
          break

      if is_openpilot_engaged:
        sm.update(0)

        # TODO gas and brake is deprecated
        op.throttle = sm['carControl'].actuators.accel
        op.brake = sm['carControl'].actuators.accel * -1
        op.steer = sm['carControl'].actuators.steeringAngleDeg
        new = TBS_scale_clamp(op, 'openpilot2carla')
        out = TBS_rate_limit(old, new, 'openpilot')
        old = out
      else:
        new = TBS_scale_clamp(manual, 'manual2carla')
        out = TBS_rate_limit(old, new, 'manual')
        old = out

      # print("prev:", old, "op:", op, "manual:", manual, "new:", new, "out", out)

      # --------------Step 2-------------------------------
      
      vc.throttle = out.throttle
      vc.brake = out.brake
      vc.steer = out.steer

      vehicle.apply_control(vc)

      # --------------Step 3-------------------------------
      vel = vehicle.get_velocity()
      speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # in m/s
      vehicle_state.speed = speed
      vehicle_state.vel = vel
      vehicle_state.angle = out.steer
      vehicle_state.cruise_button = cruise_button
      vehicle_state.is_engaged = is_openpilot_engaged

      if rk.frame % PRINT_DECIMATION == 0:
        print("frame: ", "engaged:", is_openpilot_engaged, "; throttle: ", round(vc.throttle, 3), "; steer(c/deg): ",
              round(vc.steer, 3), round(out.steer, 3), "; brake: ", round(vc.brake, 3))

      if rk.frame % 5 == 0:
        world.tick()
      rk.keep_time()
      self.started = True

  def close(self):
    self.started = False
    self._exit_event.set()

    for s in self._carla_objects:
      try:
        s.destroy()
      except Exception as e:
        print("Failed to destroy carla object", e)
    for t in reversed(self._threads):
      t.join()

  def run(self, queue, retries=-1):
    bridge_p = Process(target=self.bridge_keep_alive, args=(queue, retries), daemon=True)
    bridge_p.start()
    return bridge_p


if __name__ == "__main__":
  q: Any = Queue()
  args = parse_args()

  try:
    carla_bridge = CarlaBridge(args)
    p = carla_bridge.run(q)

    if args.joystick:
      # start input poll for joystick
      from tools.sim.lib.manual_ctrl import wheel_poll_thread

      wheel_poll_thread(q)
    else:
      # start input poll for keyboard
      from tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

      keyboard_poll_thread(q)
    p.join()

  finally:
    # Try cleaning up the wide camera param
    # in case users want to use replay after
    Params().remove("WideCameraOnly")

