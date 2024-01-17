#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from openpilot.common.realtime import Ratekeeper
from openpilot.common.retry import retry
from openpilot.system.assistant.openwakeword import Model
from openpilot.system.assistant.openwakeword.utils import download_models
from openpilot.common.params import Params
from cereal import messaging


RATE = 12.5
SAMPLE_RATE = 16000
SAMPLE_BUFFER = 1280 # (approx 100ms)


class WakeWordListener:
  def __init__(self, model):
    model_path = Path(__file__).parent / f'models/{model}.onnx'
    melspec_model_path = Path(__file__).parent / 'models/melspectrogram.onnx'
    embedding_model_path = Path(__file__).parent / 'models/embedding_model.onnx'
    self.owwModel = Model(wakeword_models=[model_path], melspec_model_path=melspec_model_path, embedding_model_path=embedding_model_path)
    self.params = Params()
    self.sm = messaging.SubMaster(['microphoneRaw'])


  def update(self):
    self.owwModel.predict(np.frombuffer(self.sm['microphoneRaw'].rawSample, dtype=np.int16))
    for mdl in self.owwModel.prediction_buffer.keys():
        scores = list(self.owwModel.prediction_buffer[mdl])
        detected = scores[-1] >= 0.5
        #curr_score = "{:.20f}".format(abs(scores[-1]))
    #print(curr_score)
    if detected:
      print("wake word detected")
    self.params.put_bool("WakeWordDetected", detected) 
    
    return detected

  def wake_word_listener_thread(self):
    while True:
        self.sm.update(0)
        if self.sm.updated['microphoneRaw']:
            print(self.sm['microphoneRaw'].frameIndex)
            self.update()

def main():
  model = "alexa_v0.1"
  download_models([model], Path(__file__).parent / 'models')
  wwl = WakeWordListener(model)
  wwl.wake_word_listener_thread()

if __name__ == "__main__":
  main()
