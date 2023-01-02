## Wing Telemetry Overlay

A tool to add telemetry overlay on top of the video files. Features:
 - OpenTX telemetry support
 - Arbitrary video files
 - Videos with missing frames (eg. DJI Goggles)
 - Mini Map

### Examples

![Sample 2](https://github.com/bakwc/wing_telemetry_overlay/raw/main/sample2.gif "Sample 2")

![Sample 1](https://github.com/bakwc/wing_telemetry_overlay/raw/main/sample1.gif "Sample 1")

### Installation

1) Download and install [python3](https://www.python.org/downloads/)
2) Download and install [pip](https://pip.pypa.io/en/stable/installation/)
3) Clone repo: `git clone https://github.com/bakwc/wing_telemetry_overlay/`
4) Install requirements: `python3 -m pip install -r requirements.txt `

### Usage

You will need:
 - Video file, eg. `my_video.mp4`
 - Telemetry data from your opentx radio (csv file, eg. `Model01-2000-07-16.csv`)
 - Fill the settings.ini file with the correct settings (see settings section below)
 - Run `python3 video_processor.py` command to generate video. It will write result to `my_video_out.mp4`. You can use keys `A`, `D`, `Z`, `C` and `Q` to control playback.

#### Main settings
Open a `settings.ini` file and set the following parameters:

 - `video_file` - path to a video file, eg. `C:\videos\my_video.mp4`
 - `sync_video_start` - a time on video (hh:mm:ss) when you launched your plane (open a video and watch it to find out the exact moment)
 - `sync_video_start` - a time on video (hh:mm:ss) when you landed your plane and it stopped moving
 - `telemetry_file` - path to opentx telemetry file, eg. `C:\videos\Model01-2000-07-16.csv`
 - `sync_telemetry_start` - a time inside telemetry file (hh:mm:ss.ms) when you launched the plane (open a telemetry file and find the first row with `GSpd(kmh)` more than 5-10 km/h - that means the plane starts moving)
 - `sync_telemetry_finish` - a time inside telemetry file (hh:mm:ss.ms) when you landed the plane (open a telemetry file and find the first row with `GSpd(kmh)` becomes zero after it was non-zero)

#### Flight modes

You need to set your controller channels, values and flight modes. You should add all modes you used in your flight. Example:
```
mode_0_name = MANUAL
mode_0_field = SC
mode_0_value = -1
```
Mode name is `MANUAL`, it's turned on by channel `SC` when you set your switch to `-1` position (there is 3 possible positions, `-1`, `0` and `1`). Another example:
```
mode_3_name = RETURN TO HOME
mode_3_field = SD
mode_3_value = 1
```
Mode name is `RETURN TO HOME`, it's turned on by channel `SD` when you set the switch to `1` position.

You can check your switch positions inside telemetry files, there are columns named `SA`, `SB`, `SC`, etc.

#### Minimap

There is a `mapbox_token` parameter. You can try to use current token. If you hit the limits - you will need to [create a mabox account](https://account.mapbox.com/auth/signup/) and [create your own token](https://account.mapbox.com/access-tokens/create).