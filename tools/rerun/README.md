# Rerun
Rerun is a tool to quickly visualize time series data. It supports openpilot logging system, both the `logMessages` and video logs.

## Usage
```
usage: run.py [-h] [--demo] [--qcam] [--fcam] [--ecam] [--dcam] [--print_services] [--services [SERVICES ...]] [route_or_segment_name]

A helper to run rerun on openpilot routes

options:
  -h, --help                  show this help message and exit
  --demo                      Use the demo route instead of providing one (default: False)
  --qcam                      Log decimated driving camera (default: False)
  --fcam                      Log driving camera (default: False)
  --ecam                      Log wide camera (default: False)
  --dcam                      Log driver monitoring camera (default: False)
  --print_services            List out openpilot services (default: False)
  --services [SERVICES ...]   Specify openpilot services that will be logged. No service will be logged if not specified.
                              To log all services include 'all' as one of your services (default: [])
  --route [ROUTE]             The route or segment name to plot (default: None)
```

Examples using route name to observe accelerometer and qcamera:

`./run.py --services accelerometer --qcam --route "a2a0ccea32023010/2023-07-27--13-01-19"`

Examples using segment range (more on [SegmentRange](https://github.com/commaai/openpilot/tree/master/tools/lib)):

`./run.py --qcam --route "a2a0ccea32023010/2023-07-27--13-01-19/2:6:2"`

## Openpilot services
To list all openpilot services:

`./run.py --print_services`

## Demo
`./run.py --services accelerometer --qcam --demo`

## Cautions:
- You can specify `--services all` to visualize all `logMessage`, but it will draw a lot of memory usage and take a long time to log all messages. Rerun isn't ready for logging big number of data.

- Logging hevc videos (`--fcam`, `--ecam`, and `--dcam`)  are expensive, and it's recommended to use `--qcam` for optimized performance. If possible, limiting your route to a few segments using `SegmentRange` will speed up logging and reduce memory usage
