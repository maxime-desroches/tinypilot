import asyncio
import json
import logging
import os
import ssl
import uuid
import pyaudio
import io
import wave

import time
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VideoStreamTrack, AudioStreamTrack, MediaStreamError
from aiortc.contrib.media import MediaBlackhole
from aiortc.rtcrtpsender import RTCRtpSender

import numpy as np
from av import VideoFrame, CodecContext, Packet
from pydub import AudioSegment
import cereal.messaging as messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from system.camerad.snapshot.snapshot import get_yuv


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

pcs = set()

IMG_H, IMG_W = 604, 964
yuv = np.zeros((int(IMG_H * 1.5), IMG_W), dtype=np.uint8)
AUDIO_RATE = 16000
pm = messaging.PubMaster(['testJoystick'])
sm = messaging.SubMaster(['carState'])
SOUNDS = {
  'engage': '../../selfdrive/assets/sounds/engage.wav',
  'disengage': '../../selfdrive/assets/sounds/disengage.wav',
  'error': '../../selfdrive/assets/sounds/warning_immediate.wav',
}


def force_codec(pc, sender, forced_codec='video/VP9', stream_type="video"):
  codecs = RTCRtpSender.getCapabilities(stream_type).codecs
  codec = [codec for codec in codecs if codec.mimeType == forced_codec]
  transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
  print("transceiver", transceiver, codec)
  transceiver.setCodecPreferences(codec)


def yuv_from_buffer(yuv_img_raw, width, height, stride, uv_offset):
  y, u, v = get_yuv(yuv_img_raw, width, height, stride, uv_offset)
  ul = np.repeat(np.repeat(u, 2).reshape(u.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
  vl = np.repeat(np.repeat(v, 2).reshape(v.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
  yuv = np.dstack((y, ul, vl)).astype(np.int16)
  yuv[:, :, 1:] -= 128
  return yuv


class VideoTrack(VideoStreamTrack):
  def __init__(self):
    super().__init__()
    self.vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
    self.cnt = 0

  async def recv(self):
    if os.environ.get('FAKE_CAMERA') == '1':
      frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)

    else:
      if not self.vipc_client.is_connected():
        self.vipc_client.connect(True)
      yuv_img_raw = self.vipc_client.recv()
      if yuv_img_raw is None or not yuv_img_raw.any():
        frame = VideoFrame.from_ndarray(np.zeros((int(self.vipc_client.height * 1.5), self.vipc_client.width), np.uint8), format="yuv420p")
      else:
        y, u, v = get_yuv(yuv_img_raw, self.vipc_client.width, self.vipc_client.height, self.vipc_client.stride, self.vipc_client.uv_offset)
        u = u.reshape(u.shape[0] // 2, -1)
        v = v.reshape(v.shape[0] // 2, -1)
        yuv[:IMG_H, :] = y[::2, ::2]
        yuv[IMG_H:int(IMG_H * 5 / 4), :] = u[::2, ::2]
        yuv[int(5 * IMG_H / 4):, :] = v[::2, ::2]
        frame = VideoFrame.from_ndarray(yuv, format="yuv420p")

    pts, time_base = await self.next_timestamp()
    frame.pts = pts
    frame.time_base = time_base
    return frame


class LiveSpeaker(MediaBlackhole):
  def __init__(self):
    super().__init__()
    self.p = pyaudio.PyAudio()
    self.buffer = io.BytesIO()
    self.channels = 2
    self.stream = self.p.open(format=pyaudio.paInt16, channels=self.channels, rate=48000, frames_per_buffer=9600, output=True, stream_callback=self.pyaudio_callback)

  def pyaudio_callback(self, in_data, frame_count, time_info, status):
    # print(self.buffer.getbuffer().nbytes)
    if self.buffer.getbuffer().nbytes < frame_count * self.channels * 2:
      buff = np.zeros((frame_count, 2), dtype=np.int16).tobytes()
    elif self.buffer.getbuffer().nbytes > 115200:  # 3x the usual read size
      self.buffer.seek(0)
      buff = self.buffer.read(frame_count * self.channels * 4)
      buff = buff[:frame_count * self.channels * 2]
      self.buffer.seek(2)
    else:
      self.buffer.seek(0)
      buff = self.buffer.read(frame_count * self.channels * 2)
      self.buffer.seek(2)
    return (buff, pyaudio.paContinue)

  async def consume(self, track):
    while True:
      try:
        frame = await track.recv()
      except MediaStreamError:
        return
      bio = bytes(frame.planes[0])
      self.buffer.write(bio)

  async def start(self):
    for track, task in self._MediaBlackhole__tracks.items():
      if task is None:
        self._MediaBlackhole__tracks[track] = asyncio.ensure_future(self.consume(track))

  async def stop(self):
    for task in self._MediaBlackhole__tracks.values():
      if task is not None:
        task.cancel()
    self._MediaBlackhole__tracks = {}
    self.stream.stop_stream()
    self.stream.close()
    self.p.terminate()


class MicTrack(AudioStreamTrack):
  def __init__(self):
    super().__init__()

    self.sample_rate = AUDIO_RATE
    self.AUDIO_PTIME = 0.020  # 20ms audio packetization
    self.samples = int(self.AUDIO_PTIME * self.sample_rate)
    self.FORMAT = pyaudio.paInt16
    self.CHANNELS = 2
    self.RATE = self.sample_rate
    self.CHUNK = int(AUDIO_RATE * 0.020)
    self.p = pyaudio.PyAudio()
    self.mic_stream = self.p.open(format=self.FORMAT, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

    self.codec = CodecContext.create('pcm_s16le', 'r')
    self.codec.sample_rate = self.RATE
    self.codec.channels = 2
    self.audio_samples = 0
    self.chunk_number = 0

  async def recv(self):
    mic_data = self.mic_stream.read(self.CHUNK)
    mic_sound = AudioSegment(mic_data, sample_width=2, channels=1, frame_rate=self.RATE)
    mic_sound = AudioSegment.from_mono_audiosegments(mic_sound, mic_sound)
    mic_sound += 3  # increase volume by 3db
    packet = Packet(mic_sound.raw_data)
    frame = self.codec.decode(packet)[0]
    frame.pts = self.audio_samples
    self.audio_samples += frame.samples
    self.chunk_number = self.chunk_number + 1
    return frame


async def play_sound(sound):
  chunk = 5120
  print("playing", sound)
  with wave.open(SOUNDS[sound], 'rb') as wf:
    def callback(in_data, frame_count, time_info, status):
      data = wf.readframes(frame_count)
      return data, pyaudio.paContinue

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=chunk,
                    stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
      await asyncio.sleep(0)
    stream.stop_stream()
    stream.close()
    p.terminate()


async def index(request):
  content = open("./static/index.html", "r").read()
  request.app['mutable_vals']['last_send_time'] = time.monotonic()
  return web.Response(content_type="text/html", text=content)


async def dummy_controls_msg(app):
  while True:
    if 'remote_channel' in app['mutable_vals']:
      this_time = time.monotonic()
      if (app['mutable_vals']['last_send_time'] + 0.5) < this_time:
        dat = messaging.new_message('testJoystick')
        dat.testJoystick.axes = [0, 0]
        dat.testJoystick.buttons = [False]
        pm.send('testJoystick', dat)
    await asyncio.sleep(0.1)


async def start_background_tasks(app):
  app['bgtask_dummy_controls_msg'] = asyncio.create_task(dummy_controls_msg(app))


async def stop_background_tasks(app):
  app['bgtask_dummy_controls_msg'].cancel()
  await app['bgtask_dummy_controls_msg']


async def control_body(data):
  print(data)
  x = max(-1.0, min(1.0, data['x']))
  y = max(-1.0, min(1.0, data['y']))
  dat = messaging.new_message('testJoystick')
  dat.testJoystick.axes = [x, y]
  dat.testJoystick.buttons = [False]
  pm.send('testJoystick', dat)


async def offer(request):
  logger.info("\n\n\nnewoffer!\n\n")
  params = await request.json()
  offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
  speaker = LiveSpeaker()

  pc = RTCPeerConnection()
  pc_id = "PeerConnection(%s)" % uuid.uuid4()
  pcs.add(pc)

  def log_info(msg, *args):
    logger.info(pc_id + " " + msg, *args)

  log_info("Created for %s", request.remote)

  @pc.on("datachannel")
  def on_datachannel(channel):
    request.app['mutable_vals']['remote_channel'] = channel

    @channel.on("message")
    async def on_message(message):
      data = json.loads(message)
      if data['type'] == 'control_command':
        await control_body(data)
        request.app['mutable_vals']['last_send_time'] = time.monotonic()
        times = {
          'type': 'ping_time',
          'incoming_time': data['dt'],
          'outgoing_time': int(time.time() * 1000),
        }
        channel.send(json.dumps(times))
      if data['type'] == 'battery_level':
        sm.update(timeout=50)
        if sm.updated['carState']:
          channel.send(json.dumps({'type': 'battery_level', 'value': int(sm['carState'].fuelGauge * 100)}))
      if data['type'] == 'play_sound':
        await play_sound(data['sound'])

  @pc.on("connectionstatechange")
  async def on_connectionstatechange():
    log_info("Connection state is %s", pc.connectionState)
    if pc.connectionState == "failed":
      await pc.close()
      pcs.discard(pc)

  @pc.on('track')
  def on_track(track):
    print("Track received")
    speaker.addTrack(track)

    @track.on("ended")
    async def on_ended():
      log_info("Remote %s track ended", track.kind)
      await speaker.stop()

  video_sender = pc.addTrack(VideoTrack())
  force_codec(pc, video_sender)
  _ = pc.addTrack(MicTrack())

  await pc.setRemoteDescription(offer)
  await speaker.start()
  answer = await pc.createAnswer()
  await pc.setLocalDescription(answer)

  return web.Response(
    content_type="application/json",
    text=json.dumps(
      {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    ),
  )


async def on_shutdown(app):
  coros = [pc.close() for pc in pcs]
  await asyncio.gather(*coros)
  pcs.clear()


async def run(cmd):
  proc = await asyncio.create_subprocess_shell(
    cmd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
  )
  stdout, stderr = await proc.communicate()
  print("Created key and cert!")
  if stdout:
    print(f'[stdout]\n{stdout.decode()}')
  if stderr:
    print(f'[stderr]\n{stderr.decode()}')


if __name__ == "__main__":
  # App needs to be HTTPS for microphone and audio autoplay to work on the browser
  if (not os.path.exists("cert.pem")) or (not os.path.exists("key.pem")):
    asyncio.run(run('openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/C=US/ST=California/O=commaai/OU=comma body"'))
  else:
    print("Certificate exists!")
  ssl_context = ssl.SSLContext()
  ssl_context.load_cert_chain("cert.pem", "key.pem")
  app = web.Application()
  app['mutable_vals'] = {}
  app.on_shutdown.append(on_shutdown)
  app.router.add_post("/offer", offer)
  app.router.add_get("/", index)
  app.router.add_static('/static', 'static')
  app.on_startup.append(start_background_tasks)
  app.on_cleanup.append(stop_background_tasks)
  web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)
