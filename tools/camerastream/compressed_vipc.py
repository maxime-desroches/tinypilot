#!/usr/bin/env python3
import os
import sys
import numpy as np
import multiprocessing

from cereal.visionipc.visionipc_pyx import VisionIpcServer, VisionStreamType  # pylint: disable=no-name-in-module, import-error
W, H = 1928, 1208

V4L2_BUF_FLAG_KEYFRAME = 8

def writer(fn, addr, sock_name):
  import cereal.messaging as messaging
  fifo_file = open(fn, "wb")

  os.environ["ZMQ"] = "1"
  messaging.context = messaging.Context()

  sock = messaging.sub_sock(sock_name, None, addr=addr, conflate=False)
  last_idx = -1
  seen_iframe = False
  while 1:
    msgs = messaging.drain_sock(sock, wait_for_one=True)
    for evt in msgs:
      evta = getattr(evt, evt.which())
      lat = ((evt.logMonoTime/1e9) - (evta.idx.timestampEof/1e9))*1000
      print("%2d %4d %.3f %.3f latency %.2f ms" % (len(msgs), evta.idx.encodeId, evt.logMonoTime/1e9, evta.idx.timestampEof/1e6, lat), len(evta.data), sock_name)
      if evta.idx.encodeId != 0 and evta.idx.encodeId != (last_idx+1):
        print("DROP!")
      last_idx = evta.idx.encodeId
      if evta.idx.flags & V4L2_BUF_FLAG_KEYFRAME:
        fifo_file.write(evta.header)
        seen_iframe = True
      if not seen_iframe:
        print("waiting for iframe")
        continue
      fifo_file.write(evta.data)

FFMPEG_OPTIONS = {"probesize": "32", "flags": "low_delay"}

def decoder_nvidia(fn, vipc_server, vst, yuv=True, rgb=False):
  sys.path.append("/raid.dell2/PyNvCodec")
  import PyNvCodec as nvc # pylint: disable=import-error
  decoder = nvc.PyNvDecoder(fn, 0, FFMPEG_OPTIONS)
  cc1 = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709, nvc.ColorRange.JPEG)

  if rgb:
    conv = nvc.PySurfaceConverter(W, H, nvc.PixelFormat.NV12, nvc.PixelFormat.BGR, 0)
    nvDwn = nvc.PySurfaceDownloader(W, H, nvc.PixelFormat.BGR, 0)
    img = np.ndarray((H,W,3), dtype=np.uint8)

  if yuv:
    conv_yuv = nvc.PySurfaceConverter(W, H, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, 0)
    nvDwn_yuv = nvc.PySurfaceDownloader(W, H, nvc.PixelFormat.YUV420, 0)
    img_yuv = np.ndarray((H*W//2*3), dtype=np.uint8)

  cnt = 0
  while 1:
    rawSurface = decoder.DecodeSingleSurface()
    if rawSurface.Empty():
      continue
    if rgb:
      convSurface = conv.Execute(rawSurface, cc1)
      nvDwn.DownloadSingleSurface(convSurface, img)
      vipc_server.send(vst, img.flatten().data, cnt, 0, 0)
    if yuv:
      convSurface = conv_yuv.Execute(rawSurface, cc1)
      nvDwn_yuv.DownloadSingleSurface(convSurface, img_yuv)
      vipc_server.send(vst+3, img_yuv.flatten().data, cnt, 0, 0)
    cnt += 1

def decoder_ffmpeg(fn, vipc_server, vst, yuv=True, rgb=False):
  import av # pylint: disable=import-error
  container = av.open(fn, options=FFMPEG_OPTIONS)
  cnt = 0
  for frame in container.decode(video=0):
    if rgb:
      img = frame.to_ndarray(format=av.video.format.VideoFormat('bgr24'))
      vipc_server.send(vst, img.flatten().data, cnt, 0, 0)
    if yuv:
      img_yuv = frame.to_ndarray(format=av.video.format.VideoFormat('yuv420p'))
      vipc_server.send(vst+3, img_yuv.flatten().data, cnt, 0, 0)
    cnt += 1

import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Decode video streams and broacast on VisionIPC')
  parser.add_argument("addr", help="Address of comma 3")
  parser.add_argument('--pipes', action='store_true', help='Only create pipes')
  parser.add_argument('--nvidia', action='store_true', help='Use nvidia instead of ffmpeg')
  parser.add_argument('--rgb', action='store_true', help='Also broadcast RGB')
  parser.add_argument("--cams", default="0,1,2", help="Cameras to decode")
  args = parser.parse_args()

  all_cams = [
    ("roadEncodeData", VisionStreamType.VISION_STREAM_RGB_ROAD),
    ("wideRoadEncodeData", VisionStreamType.VISION_STREAM_RGB_WIDE_ROAD),
    ("driverEncodeData", VisionStreamType.VISION_STREAM_RGB_DRIVER),
  ]
  cams = dict([all_cams[int(x)] for x in args.cams.split(",")])

  vipc_server = VisionIpcServer("camerad")
  for vst in cams.values():
    if args.rgb:
      vipc_server.create_buffers(vst, 4, True, W, H)
    vipc_server.create_buffers(vst+3, 4, False, W, H)
  vipc_server.start_listener()

  for k,v in cams.items():
    FIFO_NAME = "/tmp/decodepipe_"+k
    if os.path.exists(FIFO_NAME):
      os.unlink(FIFO_NAME)
    os.mkfifo(FIFO_NAME)
    multiprocessing.Process(target=writer, args=(FIFO_NAME, sys.argv[1], k)).start()
    if args.pipes:
      print("connect to", FIFO_NAME)
    elif args.nvidia:
      multiprocessing.Process(target=decoder_nvidia, args=(FIFO_NAME, vipc_server, v, True, args.rgb)).start()
    else:
      multiprocessing.Process(target=decoder_ffmpeg, args=(FIFO_NAME, vipc_server, v, True, args.rgb)).start()
