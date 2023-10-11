import aiortc
import asyncio
import aiohttp

import abc
import dataclasses
import json


@dataclasses.dataclass
class StreamingOffer:
  sdp: str
  type: str
  video: list[str]
  audio: bool


class ConnectionProvider(abc.ABC):
  async def connect(self, offer) -> aiortc.RTCSessionDescription:
    raise NotImplementedError()


class StdioConnectionProvider(ConnectionProvider):
  async def connect(self, offer: StreamingOffer) -> aiortc.RTCSessionDescription:
    async def async_input():
      return await asyncio.to_thread(input)

    print("-- Please send this JSON to server --")
    print(json.dumps(dataclasses.asdict(offer)))
    print("-- Press enter when the answer is ready --")
    raw_payload = await async_input()
    payload = json.loads(raw_payload)
    answer = aiortc.RTCSessionDescription(**payload)

    return answer


class HTTPConnectionProvider(ConnectionProvider):
  def __init__(self, address="127.0.0.1", port=8080):
    self.address = address
    self.port = port

  async def connect(self, offer: StreamingOffer) -> aiortc.RTCSessionDescription:
    payload = dataclasses.asdict(offer)
    async with aiohttp.ClientSession() as session:
      response = await session.get(f"http://{self.address}:{self.port}/webrtc", json=payload)
      async with response:
        if response.status != 200:
          raise Exception(f"Offer request failed with HTTP status code {response.status}")
        answer = await response.json()
        remote_offer = aiortc.RTCSessionDescription(sdp=answer.sdp, type=answer.type)

        return remote_offer
