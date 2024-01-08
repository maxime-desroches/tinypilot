#!/usr/bin/env python3
import threading
import os

from cereal.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.webcam.camera import Camera

YUV_BUFFER_COUNT = int(os.getenv("YUV_BUFFER_COUNT", "20"))
CAMERA_TYPES ={"road_cam":VisionStreamType.VISION_STREAM_ROAD,
               "driver_cam":VisionStreamType.VISION_STREAM_DRIVER,
               "wide_cam":VisionStreamType.VISION_STREAM_DRIVER
              }

class Camerad:
  def __init__(self):
    self.pm = messaging.PubMaster(['roadCameraState', 'driverCameraState', 'wideRoadCameraState'])
    self.vipc_server = VisionIpcServer("camerad")

    self.frame_road_id, self.frame_driver_id, self.frame_wide_id = 0, 0, 0
    self.dual_camera = bool(int(os.getenv("DUAL","0")))
    self.cameras, self.camera_threads = [], [] # ORDER: road_cam, driver_cam, wide_cam

    for cam_type, stream_type in CAMERA_TYPES.items():
      if cam_type == "wide_cam":
        if not self.dual_camera: 
          break
        cam = Camera(cam_type, int(os.getenv("CAMERA_WIDE_ID", "2")))
      elif cam_type == "road_cam":
        cam = Camera(cam_type, int(os.getenv("CAMERA_ROAD_ID", "0")))
      else:
        #cam = Camera(cam_type, int(os.getenv("CAMERA_DRIVER_ID", "1")))
        cam = Camera(cam_type, "/home/bongb/unlearn/calib_challenge/labeled/0.hevc")
      assert cam.cap.isOpened(), f"Can't find {cam_type}"
      self.cameras.append(cam)
      self.vipc_server.create_buffers(stream_type, YUV_BUFFER_COUNT, False, cam.W, cam.H)

    self.vipc_server.start_listener()

  #from sim
  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    eof = int(frame_id * 0.05 * 1e9)
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)
    dat = messaging.new_message(pub_type, valid=True)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)

  @classmethod
  def daemon_alive(self, cam, send_yuv):
    #while cam.cap.isOpened():
    while True:
      for yuv in cam.read_frames():
        send_yuv(yuv)

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_driver(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'driverCameraState', VisionStreamType.VISION_STREAM_DRIVER)
    self.frame_driver_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1
  
  def start_camera_threads(self):
    for cam in self.cameras:
      camera_type = cam.cam_type
      if camera_type == "wide_cam":
        if not self.dual_camera: 
          break
        cam_thread = threading.Thread(target=Camerad.daemon_alive, args=(cam, self.cam_send_yuv_wide_road))
      elif camera_type == "road_cam":
        cam_thread = threading.Thread(target=Camerad.daemon_alive, args=(cam, self.cam_send_yuv_road))
      else:
        cam_thread = threading.Thread(target=Camerad.daemon_alive, args=(cam, self.cam_send_yuv_driver))
      print(f"STARTING {camera_type}")
      cam_thread.start()
      self.camera_threads.append(cam_thread)
  
  def join_camera_threads(self):
    for thread in self.camera_threads:
      thread.join()

  def run(self):
    self.start_camera_threads()
    self.join_camera_threads()

if __name__ == "__main__":
  camerad = Camerad()
  camerad.run()
