# ROS 
## 0. Folder Config:
- `merlion_all`: 
  - `base.launch`: to launch all components (`pid_publisher`, `state_publisher`, `all_trackers`, `all_hardware`)
  - Usage: `roslaunch merlion_all base.launch`
- `merlion_cv`: to put your cv code ***(TODO)***
  - `all_trackers.launch`: to launch both trackers (`flare_tracker.launch` and `gate_tracker.launch`)
  - `flare_tracker.launch`: CV strategy to track flare using `merlion_cv/scripts/flare_tracker.py`
  - `gate_tracker.launch`: CV strategy to track gate using `merlion_cv/scripts/gate_tracker.py`
- `merlion_hardware`: to put code for your imu, mcu, cameras
  - `arduino/mcu`: deprecated, can't refer for seniors code
  - `msg`: define our custom message type for 5 motors 
  - `all_hardware.launch`: to activate all hardware (`mcu.launch`, `cameras.launch`, `imu.launch`)
  - `cameras.launch`: activate your camera with openCV (***TODO***)
  - `imu.launch`: activate IMU port (***TODO***)
  - `mcu.launch`: activate MCU prt, connect to our motors/thrusters under `merlion_hardware/scripts/thruster_manager.py`
  - `localization.launch`: include `optical_odometer.launch` (***TODO***)
- `merlion_pid`: our pid publisher
  - `pid_publisher.launch`: to launch pid using `merlion_pid/scripts/pid_publisher.py` and `merlion_pid/scripts/setpoint_once.py`
- `merlion_sensors/vectornav`: (***TO BE DOCUMENTED BY CLARENCE***) 
- `merlion_state`: our state publisher
  - `state_publisher.launch`: activate state publisher with `state_publisher.py`

## 1. Current Status
:::info
We are currently still under development phase so we only use `merlion_hardware/mcu.launch`, `merlion_pid`, `merlion_state` and `merlion_sensors/vectornav`
:::
- To launch (***TO BE DOCUMENTED BY CLARENCE***) 
```
# Terminal 1:
roscore

# Terminal 2:
.......
```
- Current graph (***RQT GRAPH TO BE INSERTED BY CLARENCE***)

## 2. TODO
-  Add cv strategy to `merlion_cv`
-  Figure out what `localization.launch` is for 

## 3. Deprecated
### This folder will contain the ROS main architecture as suggested by Blong. All is adapted from the seniors code.
### The imu/sensor input and cv input will be done by nav team
### The thruster output will be done by Pin
![merlion_cv merlion_state](https://user-images.githubusercontent.com/65146336/118364574-8e943c00-b5cb-11eb-9e13-78975cb2dd0d.png)
![merlion_pid](https://user-images.githubusercontent.com/65146336/118364578-93f18680-b5cb-11eb-8177-26ae82c754dc.png)
![merlion_thruster](https://user-images.githubusercontent.com/65146336/118364584-9653e080-b5cb-11eb-9e70-13eca1cfd141.png)

