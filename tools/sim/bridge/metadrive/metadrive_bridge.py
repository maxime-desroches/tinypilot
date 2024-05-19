import math

from collections import namedtuple
from multiprocessing import Queue

from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }

def curve_block(length, angle=45, direction=0):
  return {
    "id": "C",
    "pre_block_socket_index": 0,
    "length": length,
    "radius": length,
    "angle": angle,
    "dir": direction
  }

def create_map(track_size=60):
  curve_len = track_size * 2 + 10 # empirically reliable for driving test
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,
    lane_width=4.5,
    config=[
      None,
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
    ]
  )

done_config = namedtuple("done_config", ["out_of_route_done", "on_continuous_line_done"], defaults=[False, False])


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, time_done=math.inf, test_run=False):
    self.should_render = False
    self.test_run = test_run
    self.time_done = time_done if self.test_run else math.inf
    self.done_config = done_config(True, True) if self.test_run else done_config()

    super().__init__(dual_camera, high_quality)

  def spawn_world(self, queue: Queue):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H, )
    }

    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    config = dict(
      use_render=self.should_render,
      vehicle_config=dict(
        enable_reverse=False,
        image_source="rgb_road",
      ),
      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      out_of_route_done=self.done_config.out_of_route_done,
      on_continuous_line_done=self.done_config.on_continuous_line_done,
      crash_vehicle_done=False,
      crash_object_done=False,
      arrive_dest_done=False,
      traffic_density=0.0, # traffic is incredibly expensive
      map_config=create_map(),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False
    )

    return MetaDriveWorld(queue, config, self.time_done, self.test_run, self.dual_camera)
