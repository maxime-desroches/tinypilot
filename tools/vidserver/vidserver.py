#!/usr/bin/env python3
import json
import os
import subprocess
import time
import threading

from flask import Flask, send_from_directory, Response, request
from selfdrive.loggerd.config import ROOT
from selfdrive.loggerd.uploader import listdir_by_creation
from tools.lib.route import SegmentName

def all_segment_names(root=ROOT):
  fake_dongle = "ffffffffffffffff"
  segments = []
  for segment in listdir_by_creation(root):
    try:
      absolute_path = str(os.path.join(root, fake_dongle + "|" + segment))
      segments.append(SegmentName(absolute_path))
    except:
      pass
  return segments

def all_routes():
  segment_names = all_segment_names()
  route_names = [segment_name.route_name for segment_name in segment_names]
  route_times = [route_name.time_str for route_name in route_names]
  unique_routes = list(set(route_times))
  return unique_routes

def segments_in_route(route):
  segment_names = [segment_name for segment_name in all_segment_names() if segment_name.time_str == route]
  segments = [segment_name.time_str + "--" + str(segment_name.segment_num) for segment_name in segment_names]
  return segments

routes = all_routes()
for route in routes:
  print(route)
  print(segments_in_route(route))

exit()

dir_server = "/data/openpilot/tools/vidserver/"
dir_media = "/data/media/0/realdata/"

app = Flask(__name__,)

@app.route("/<type>/<segment>")
def fcamera(type, segment):
  if type in ['fcamera', 'dcamera', 'ecamera']:
    proc = subprocess.Popen(
      ["ffmpeg",
        "-f", "hevc",
        "-r", "20",
        "-i", dir_media + segment + "/" + type + ".hevc",
        "-c", "copy",
        "-map", "0",
        "-vtag", "hvc1",
        "-f", "mp4",
        "-movflags", "empty_moov",
        "-",
      ], stdout=subprocess.PIPE
    )
  elif type in ['qcamera']:
    proc = subprocess.Popen(
      ["ffmpeg",
        "-r", "20",
        "-i", dir_media + segment + "/qcamera.ts",
        "-c", "copy",
        "-map", "0",
        "-f", "mp4",
        "-movflags", "empty_moov",
        "-",
      ], stdout=subprocess.PIPE
    )
  response = Response(proc.stdout.read(), status=200, mimetype='video/mp4')
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response

@app.route("/<route>")
def route(route):
  if len(route) != 20:
    return "route not found"

  if str(request.query_string) == "b''":
    query_segment = str("0")
    query_type = "qcamera"
  else:
    query_segment = (str(request.query_string).split(","))[0][2:]
    query_type = (str(request.query_string).split(","))[1][:-1]

  links = ""
  segments = ""
  for segment in get_segments():
    if route in segment:
      links += "<a href='"+route+"?"+segment.split("--")[2]+","+query_type+"'>"+segment+"</a><br>"
      segments += "'"+segment+"',"
  return """<html>
  <body>
    <video id="video" width="320" height="240" controls autoplay="autoplay" style="background:black">
    </video>
    <br><br>
    current segment: <span id="currentsegment"></span>
    <br>
    current view: <span id="currentview"></span>
    <br><br>
    <a href="\\">back to routes</a>
    <br><br>
    <a href=\""""+route+"""?0,qcamera\">qcamera</a> - 
    <a href=\""""+route+"""?0,fcamera\">fcamera</a> - 
    <a href=\""""+route+"""?0,dcamera\">dcamera</a> - 
    <a href=\""""+route+"""?0,ecamera\">ecamera</a>
    <br><br>
    """+links+"""
  </body>
    <script>
    var video = document.getElementById('video');

    var tracks = {
      list: ["""+segments+"""],
      index: """+query_segment+""",
      next: function() {
        if (this.index == this.list.length - 1) this.index = 0;
        else {
            this.index += 1;
        }
      },
      play: function() {
        return ( \""""+query_type+"""/" + this.list[this.index] );
      }
    }

    video.addEventListener('ended', function(e) {
      tracks.next();
      video.src = tracks.play();
      document.getElementById("currentsegment").textContent=video.src.split("/")[4];
      document.getElementById("currentview").textContent=video.src.split("/")[3];
      video.load();
      video.play();
    });

    video.src = tracks.play();
    document.getElementById("currentsegment").textContent=video.src.split("/")[4];
    document.getElementById("currentview").textContent=video.src.split("/")[3];
    </script>
</html>
"""

@app.route("/")
def index():
  result = ""
  for route in get_routes():
    result += "<a href='"+route+"'>"+route+"</a><br>"
  return result

def main():
  app.run(host="0.0.0.0", port="8081")

if __name__ == '__main__':
  main()
