import warnings
import numpy as np
import metadrive  # noqa: F401 pylint: disable=W0611
import gym

from tools.sim.bridge.common import World, SimulatorBridge, Camerad, STEER_RATIO, W, H

class MetaDriveWorld(World):
  def __init__(self, env, ticks_per_frame: float):
    self.env = env
    self.speed = 0.0
    self.yuv = None
    self.camerad = Camerad()
    self.ticks_per_frame = ticks_per_frame

  def apply_controls(self, steer_sim, throttle_out, brake_out, rk):
    vc = [0.0, 0.0]
    vc[0] = steer_sim * -1
    if throttle_out:
      vc[1] = throttle_out * 10
    else:
      vc[1] = -brake_out
    if rk.frame % self.ticks_per_frame == 0:
      if (rk.frame % self.ticks_per_frame * 2) == 0:
        o, _, d, _ = self.env.step(vc)
        if d:
          print("!!!Episode terminated due to violation of safety!!!")
          self.env.reset()
          self.env.step([0.0, 0.0])
          for _ in range(300):
            rk.keep_time()
        
        self.speed = o["state"][3] * self.ticks_per_frame * 2 # empirically derived

      img = self.env.vehicle.image_sensors["rgb_wide"].get_pixels_array(self.env.vehicle, False)
      self.yuv = self.camerad.img_to_yuv(img)
      self.camerad.cam_send_yuv_wide_road(self.yuv)

  def get_velocity(self):
    return None

  def get_speed(self) -> float:
    return self.speed

  def get_steer_correction(self) -> float:
    max_steer_angle = 75 / self.ticks_per_frame
    return max_steer_angle * STEER_RATIO * -1
  
  def tick(self):
    pass


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 10

  def __init__(self, args):
    if args.dual_camera:
      warnings.warn("Dual camera not supported in MetaDrive simulator for performance reasons")
      args.dual_camera = False
    if args.ticks_per_frame:
      self.TICKS_PER_FRAME = args.ticks_per_frame
    super(MetaDriveBridge, self).__init__(args)

  def spawn_objects(self):
    env = gym.make('MetaDrive-10env-v0', config=dict(offscreen_render=True))

    # config = dict(
    #   # camera_dist=3.0,
    #   # camera_height=1.0,
    #   # use_render=True,
    #   vehicle_config=dict(
    #     # enable_reverse=True,
    #     # image_source="rgb_camera",
    #     # rgb_camera=(0,0)
    #   ),
    #   offscreen_render=True,
    # )  
    # from metadrive.utils.space import VehicleParameterSpace
    # from metadrive.component.vehicle.vehicle_type import DefaultVehicle
    # max_engine_force = VehicleParameterSpace.DEFAULT_VEHICLE["max_engine_force"]
    # max_engine_force._replace(max=max_engine_force.max * 10)
    # max_engine_force._replace(min=max_engine_force.min * 10)

    env.reset()
    from metadrive.constants import CamMask
    from metadrive.component.vehicle_module.base_camera import BaseCamera
    from metadrive.engine.engine_utils import engine_initialized
    from metadrive.engine.core.image_buffer import ImageBuffer
    class RGBCameraWide(BaseCamera):
      # shape(dim_1, dim_2)
      BUFFER_W = W  # dim 1
      BUFFER_H = H  # dim 2
      CAM_MASK = CamMask.RgbCam

      def __init__(self):
        assert engine_initialized(), "You should initialize engine before adding camera to vehicle"
        self.BUFFER_W, self.BUFFER_H = W, H
        super(RGBCameraWide, self).__init__()
        cam = self.get_cam()
        lens = self.get_lens()
        cam.lookAt(0, 2.4, 1.3)
        cam.setHpr(0, 0.8, 0)
        lens.setFov(160)
        lens.setAspectRatio(1.15)
  
    env.vehicle.add_image_sensor("rgb_wide", RGBCameraWide())

    # Monkey patch `get_rgb_array` since it involves unnecessary copies
    def patch_get_rgb_array(self):
      if self.engine.episode_step <= 1:
        self.engine.graphicsEngine.renderFrame()
      origin_img = self.cam.node().getDisplayRegion(0).getScreenshot()
      img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
      img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
      img = img[::-1]
      img = img[..., :-1]
      return img

    setattr(ImageBuffer, "get_rgb_array", patch_get_rgb_array)

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(30):
      env.step([0.0, 1.0])

    return MetaDriveWorld(env, self.TICKS_PER_FRAME)