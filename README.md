# Wolf ROS2 JAUS Bridge

ROS2 interface for the Wolf DBW vehicle with JAUS protocol support.

## Overview

This repository is a ROS 2 Humble stack for the TAMU "Wolf" drive-by-wire (DBW)
vehicle. It provides:

- A **JAUS bridge** (`wolf_jaus_bridge`) that talks to the vehicle's BeagleBone
  over UDP so the robot can be driven from a joystick and/or autonomously.
- **Waypoint navigation**: plan a path by clicking on a live satellite map in
  RViz on the driver station, send it to the vehicle, and the vehicle follows it
  using GPS + IMU.
- A **full vehicle simulation** (`fake_data`) so the entire drive flow can be
  tested on a laptop without touching the real robot.
- **VectorNav GNSS/IMU integration** (via the `vectornav` submodule) used for
  position and heading tracking.

Three machines participate, linked over Silvus radios:

| Machine | Role | Address |
|---|---|---|
| DS Pi | Driver station: RViz + path manager + Xbox | `192.168.0.52` |
| Wolf PC | Vehicle: VectorNav driver, `wolf_jaus`, pathing | `192.168.0.50` / `192.168.1.50` |
| BeagleBone | JAUS hardware bridge inside the Wolf | `192.168.1.42` (UDP `3794`) |

All machines run the same repo, ROS 2 Humble, `ROS_DOMAIN_ID=42`,
`ROS_LOCALHOST_ONLY=0`.

## Architecture

```
┌─────────────────────┐   Silvus Radio    ┌─────────────────────┐    Ethernet     ┌──────────────────┐
│  DS PI  (192.168.0.52)  │◄─(192.168.0.156)  │  WOLF PC  (192.168.0.50)  │◄───────────────►│ Wolf DBW Vehicle │
│                       │  (192.168.0.159)─►│  (192.168.1.50)          │                │  (192.168.1.42) │
│  - rosboard_node      │                   │  - vectornav driver       │                │   - PrimitiveDriver │
│  - joy_node           │                   │  - input_node             │                │   - VelocityStateS. │
│  - path_manager_node  │                   │  - drive_node             │                └──────────────────┘
│  - rviz2 (PathManager │                   │  - pathing_node           │
│    Panel + AerialMap) │                   │  - teleop_node            │
│  - Xbox controller    │                   │  - path_recorder_node     │
│                       │                   │  - map_origin_publisher   │
└───────────────────────┘                   │  - wolf_localization (EKF)│
                                            │  - robot_state_publisher  │
                                            └───────────────────────────┘
```

### Command flow

The DS Pi is the **only** source of `/joy` (the Xbox controller). It crosses the
radio to the vehicle, where the vehicle-side nodes react to it.

- `joy_node` (DS) publishes raw joystick data on `/joy`.
- `input_node` (vehicle) watches `/joy`, picks a drive mode
  (`TELEOP` / `PATHING`), requests enable/disable via the
  `request_state_change` service, and forwards the active command to `/cmd_vel`.
- `teleop_node` (vehicle) converts joystick axes to `/cmd_vel_teleop`.
- `pathing_node` (vehicle) computes steering commands toward the current
  waypoint and publishes `/cmd_vel_pathing`.
- `drive_node` (vehicle) takes `/cmd_vel` and sends wrench-effort commands to
  the BeagleBone over JAUS.
- `path_recorder_node` (vehicle) lets the controller record/save paths.
- `map_origin_publisher_node` (vehicle) locks a fixed UTM origin and publishes
  the `map` -> `base_footprint` TF used to anchor the satellite map.

## Repository Layout

```
wolf_ros/
├── docs/                      # all documentation (see below)
├── docker_ds/                 # driver station containers (repo-root build context)
│   ├── Dockerfile
│   ├── docker-compose.yml          # base (DS Pi)
│   ├── docker-compose.laptop.yml   # laptop overlay (software GL)
│   ├── docker-compose.fake.yml     # simulated vehicle for laptop testing
│   └── wolf.rviz                   # RViz config with PathManagerPanel baked in
├── docker_wolf/               # vehicle containers (repo-root build context)
│   ├── Dockerfile
│   └── docker-compose.yml          # vectornav, wolf_jaus, wolf_localization, wolf_description
├── scripts/
│   ├── run_wolf.sh                 # add 192.168.1.50 IP + bring up vehicle
│   ├── add-ip.sh                   # add second IP on the Wolf PC
│   └── run_ds.sh                   # fix .Xauthority + bring up driver station
└── src/                       # ROS 2 workspace (all packages)
    ├── wolf_ds.launch.py           # DS core: rosboard + joy + path_manager
    ├── jaus_interface/             # JAUS Python client + srv definitions
    ├── wolf_jaus_bridge/           # vehicle-side nodes
    ├── path_manager/               # DS waypoint manager (Python package)
    ├── wolf_rviz_control/          # DS RViz panel plugin (C++)
    ├── fake_data/                  # simulated vehicle (Python package)
    ├── wolf_localization/          # robot_localization EKF (not currently used)
    ├── wolf_encoder_odom/          # wheel encoder odometry (deprecated)
    ├── wolf_description/           # robot URDF + meshes
    ├── testing/                    # jaus_cli.py, xbox_driving.py, pure_persuit_test.py
    ├── vectornav/                  # VectorNav driver (git submodule, private repo)
    └── PlatformIO/                 # encoder microcontroller firmware + logs
```

### Node inventory

| Container | Nodes | Purpose |
|---|---|---|
| `wolf_ds` | `rosboard_node`, `joy_node`, `path_manager_node` | Web topic viewer, joystick input, waypoint planning |
| `wolf_rviz` | `rviz2` (+ PathManagerPanel) | Satellite map, robot model, path preview/controls |
| `wolf_fake_data` | `fake_data_node` | Simulated vehicle (laptop only) |
| `vectornav` | `vectornav`, `vn_sensor_msgs` | Raw + standard VectorNav topics |
| `wolf_jaus` | `input_node`, `drive_node`, `pathing_node`, `teleop_node`, `path_recorder_node`, `map_origin_publisher_node` | Vehicle control stack |
| `wolf_localization` | `robot_localization` EKF | Present but **not** used by the TF tree / follower |
| `wolf_description` | `robot_state_publisher` | Publishes `/robot_description` + URDF TFs |

## Quick Start

### Clone the repo on all machines

The `src/vectornav` folder is a **git submodule** (private repo). The DS image
never copies it, so the DS Pi can skip it; the vehicle needs it.

```bash
# Vehicle (Wolf PC):
git clone --recurse-submodules git@github.com:tamu-edu/wolf_ros.git && cd wolf_ros

# DS Pi / laptop (no submodule needed):
git clone git@github.com:tamu-edu/wolf_ros.git && cd wolf_ros
```

> The `tamu-edu/wolf_ros` repo is **private**. A plain `https://` clone will
> prompt for a GitHub username/password and fail non-interactively. Use an SSH
> clone or a personal access token with https.

### Configure network settings

Configure network settings as noted in
[docs/wireless_config.md](./docs/wireless_config.md):

- DS Pi: `192.168.0.52`
- Wolf PC: `192.168.0.50` (radio) **and** `192.168.1.50` (vehicle LAN)
- BeagleBone: `192.168.1.42`

### Run start scripts

- `/scripts/run_wolf.sh` - Runs on the wolf pc (also adds the `192.168.1.50`
  secondary IP with `nmcli`)
- `/scripts/add-ip.sh` - Runs on the wolf pc to add the second IP address
- `/scripts/run_ds.sh` - Runs on the driver station pi (fixes the Xauthority
  file so the RViz window appears on the Pi desktop)

## Waypoint Navigation

The robot is tracked **entirely from the VectorNav GNSS/IMU** - there is no
encoder odometry and no EKF in the TF tree or follower.

1. `map_origin_publisher_node` (vehicle) or `fake_data_node` (laptop) publishes
   a constant UTM origin on `/nav/map_origin` at 1 Hz. It locks the origin from
   the **first sane GPS fix** (rejecting `0,0` / non-finite fixes, which the
   VectorNav reports at boot before its INS converges).
2. `path_manager_node` (DS) converts it to lat/lon and republishes it on
   `/nav/map_origin_fix` (1 Hz). The RViz AerialMap anchors to this fixed fix,
   so the satellite image stays still while the vehicle drives.
3. The operator clicks **Publish Point** on the map (fixed frame `map`). Each
   click is added to the map-origin UTM and converted to absolute lon/lat.
   Waypoints appear as axes markers on `/nav/path_manager/preview`.
4. The operator presses **Send**. `path_manager_node` publishes the lon/lat
   path on `/nav/waypoint_path` (frame `gps`).
5. The follower (`pathing_node` on the vehicle, or `fake_data_node` on the
   laptop with `motion=follow_path`) tracks the current waypoint and reports
   `/nav/path_status`: `Received a new path with N waypoints!` ->
   `Waypoint N/M reached!` -> `Path finished`. The panel shows these in its
   status box.
6. **Appending and re-Sending continues from the end.** Both followers keep
   their latched path after `Path finished`; if the new path is the old one
   with waypoints appended, they continue from their current waypoint
   (prefix-extend).
7. **Clear stops the vehicle.** `path_manager_node` wipes its waypoint buffer
   *and* publishes an empty path, so the follower stops immediately.

### Driver station (DS Pi / laptop)

```bash
cd docker_ds
docker compose build wolf_rviz        # only after editing src/ packages
docker compose up -d wolf_ds wolf_rviz
```

In the RViz window:

- **Fixed frame**: `map`
- **AerialMap**: anchored at the vehicle origin (default RELLIS BCDC area,
  `30.640567, -96.487322`)
- **RobotModel**: `wolf_chassis_link` + 6 wheels rendered on the map
- **PathManagerPanel**: `Send` / `Undo` / `Clear` / `Save` + status box
- **Publish Point** tool: click waypoints on the map

### Vehicle (Wolf PC)

```bash
cd docker_wolf
docker compose build wolf_jaus wolf_localization wolf_description
docker compose up -d
```

The `wolf_jaus` container bind-mounts `src/wolf_jaus_bridge`, so Python source
edits are live after `docker compose restart wolf_jaus` - no rebuild. The
launch file and installed executables come from the image build, so **rebuild
after changing those**.

### Laptop: full end-to-end simulation

The simulated vehicle already defaults to `follow_path`. On a laptop add the
software-GL overlay:

```bash
cd docker_ds
docker compose build          # builds the ros-humble-ds image once
docker compose -f docker-compose.yml \
               -f docker-compose.laptop.yml \
               -f docker-compose.fake.yml up -d \
               wolf_ds wolf_fake_data wolf_rviz
```

Test other motion modes without editing files (laptop only):

```bash
docker exec wolf_ds bash -c "source /opt/ros/humble/setup.bash && \
  ros2 run fake_data fake_data_node --ros-args -p motion:=circle"
```

## ROS2 Topics

### Command Topics

- `/joy` (`sensor_msgs/msg/Joy`) - Joystick input (from joy_node, DS Pi only)
- `/cmd_vel` (`geometry_msgs/msg/Twist`) - Velocity commands (input_node -> drive_node)
- `/cmd_vel_teleop` (`geometry_msgs/msg/Twist`) - Velocity commands (teleop_node -> input_node)
- `/cmd_vel_pathing` (`geometry_msgs/msg/Twist`) - Velocity commands (pathing_node -> input_node)
- `/input/mode` (`std_msgs/msg/String`) - Current drive mode (`TELEOP` / `PATHING`)
- `/input/state` (`std_msgs/msg/String`) - Current control state (`DISABLED` / `ENABLED`)

### Navigation Topics

| Topic | Type | Published by | Notes |
|---|---|---|---|
| `/nav/map_origin` | `geometry_msgs/PointStamped` | `map_origin_publisher_node` (vehicle) or `fake_data_node` (laptop) | UTM easting/northing in `point.x`/`point.y`, 1 Hz, frame `map` |
| `/nav/map_origin_fix` | `sensor_msgs/NavSatFix` | `path_manager_node` | Constant lat/lon of the map origin, anchors the AerialMap |
| `/nav/waypoint_path` | `nav_msgs/Path` | `path_manager_node` | Raw **lon in `pose.position.x`, lat in `pose.position.y`**, frame `gps` |
| `/nav/path_manager/preview` | `nav_msgs/Path` | `path_manager_node` | Waypoints in map-frame XY for RViz display |
| `/nav/path_manager/send` | `std_msgs/Empty` | PathManagerPanel | Send current waypoints as the path |
| `/nav/path_manager/undo` | `std_msgs/Empty` | PathManagerPanel | Remove last clicked waypoint |
| `/nav/path_manager/clear` | `std_msgs/Empty` | PathManagerPanel | Clear waypoints; publishes empty path so the vehicle stops |
| `/nav/path_manager/save` | `std_msgs/Empty` | PathManagerPanel | Save path (triggers vehicle save) |
| `/nav/path_manager/save_trigger` | `std_msgs/Empty` | `path_manager_node` | Vehicle saves the current path |
| `/nav/path_manager/status` | `std_msgs/String` | `path_manager_node` | Status text shown in the panel |
| `/nav/path_status` | `std_msgs/String` | `fake_data_node` or `pathing_node` | Follower progress |
| `nav/dist_error` | `std_msgs/Float32` | `pathing_node` | Distance to current waypoint |
| `nav/heading_error` | `std_msgs/Float32` | `pathing_node` | Heading error to current waypoint |
| `/clicked_point` | `geometry_msgs/PointStamped` | RViz Publish Point tool | Operator clicks, frame `map` |

### VectorNav Topics

See [docs/VECTORNAV_ROS_TOPICS.md](./docs/VECTORNAV_ROS_TOPICS.md) for the full
field-by-field reference. The most used topics are:

- `/vectornav/gnss` (`sensor_msgs/msg/NavSatFix`) - GPS fix
- `/vectornav/imu` (`sensor_msgs/msg/Imu`) - IMU / heading
- `/vectornav/imu_uncompensated` (`sensor_msgs/msg/Imu`)
- `/vectornav/pose` (`geometry_msgs/msg/PoseWithCovarianceStamped`)
- `/vectornav/raw/gps`, `/vectornav/raw/imu`, `/vectornav/raw/ins`,
  `/vectornav/raw/attitude`, etc. (`vectornav_msgs/msg/*` composite groups)
- `/vectornav/velocity_body` (`geometry_msgs/msg/TwistWithCovarianceStamped`)
- Plus `/vectornav/magnetic`, `/vectornav/pressure`, `/vectornav/temperature`,
  and the `/vectornav/time_*` time references.

### Other Topics

- `/odom` (`nav_msgs/msg/Odometry`) - Wheel encoder odometry
  (`wolf_encoder_odom`, deprecated - see below)
- `/robot_description` (`std_msgs/msg/String`) - URDF robot description
  (`wolf_description`)
- `/tf` / `/tf_static` - TF tree (`map` -> `base_footprint` -> URDF chain)

## Services

- `request_state_change` (`jaus_interface/srv/RequestStateChange`) - Used to
  request/disengage control. `input_node` calls this whenever the deadman
  switch (RT) is pressed/released.

## Joystick Configuration

Default mapping (standard Linux gamepad indices; confirm with
`ros2 topic echo /joy` on the DS Pi):

| Input | Index | Function |
|---|---|---|
| Left Stick Vertical | Axis 1 | Forward / reverse velocity |
| Right Stick Horizontal | Axis 3 | Left / right rotation |
| Right Trigger (RT) | Axis 5 | Deadman switch / enable (hold) |
| A | Button 0 | Teleop mode toggle |
| B | Button 1 | Pathing mode toggle |
| X | Button 2 | Clear path (`path_recorder_node`) |
| Y | Button 3 | Add current waypoint (`path_recorder_node`) |
| LB | Button 4 | Cycle / load path file |
| RB | Button 5 | Save path |

The vehicle only *moves* when mode = `PATHING` **and** state = `ENABLED`
(RT held). In `TELEOP` mode the vehicle follows the sticks.

## JAUS Information

- JAUS Protocol reverse-engineering notes: [docs/jaus_interface.md](./docs/jaus_interface.md)
- JAUS Python client usage: [docs/jaus_client.md](./docs/jaus_client.md)
- Manufacturer's interface spec (PDF):
  [docs/files/H038455 - Wolf v2 IOP Interface Control Document_Rev01 (1).pdf](./docs/files/H038455%20-%20Wolf%20v2%20IOP%20Interface%20Control%20Document_Rev01%20(1).pdf)

Key facts:

- The vehicle runs JAUS at `192.168.1.42:3794`. The client uses its own JAUS
  address (e.g. `1.100.1`) and talks to the vehicle component `1.2.1`.
- Control must be re-requested at ~3 Hz (it times out after 0.5 s). Drive
  (`set_wrench_effort`) at ~20 Hz for responsiveness.
- Wrench mapping: ±100% = ±6.26 m/s linear / ±4.71 rad/s rotational. Internally
  this maps to motor RPM with track width 1.30 m, wheel radius 0.33 m,
  gear ratio 30.0.
- The Wolf only advertises 9 JAUS services. `PrimitiveDriver` and
  `PowerPlantManager` are hidden from `DiscoveryService` but still work when you
  send to them directly. Battery/genset data is not available over JAUS.
- Absence of an IOP command for >300 ms triggers a motion stop. The SHC
  E-stop always works, and the stock controller can be used as an E-STOP.

## Simulation (`fake_data`)

`fake_data_node` emulates the vehicle on a laptop for end-to-end testing of the
DS without hardware. Motion modes (default `follow_path`):

- `follow_path` - drives the sent `/nav/waypoint_path`, mirroring `pathing_node`
  behavior including prefix-extend on appended paths.
- `circle` - drives a circle of radius `circle_radius` (`sim_params.yaml`).
- `none` - static vehicle publishing the map origin.

It publishes `/vectornav/gnss`, `/vectornav/imu`, `/odom`, TF, `/nav/map_origin`,
and `/nav/path_status`. Config lives in
[src/fake_data/config/sim_params.yaml](./src/fake_data/config/sim_params.yaml).

## Encoder Odometry (deprecated)

`wolf_encoder_odom` reads two wheel encoders over serial and publishes
differential-drive odometry on `/odom` with an `odom` -> `base_footprint` TF.
It is **not** used in the current live setup (the Wolf PC has no encoder
hardware - `/dev/ttyUSB0` is the VectorNav FTDI cable) and is removed from
`docker_wolf/docker-compose.yml`. Docs:
[docs/wolf_encoder_odom.md](./docs/wolf_encoder_odom.md) and
[docs/Encoders.md](./docs/Encoders.md). The microcontroller firmware and sensor
logs live under `src/PlatformIO/encoder_microcontrollers/`.

## Localization

`wolf_localization` runs `robot_localization` EKF on the vehicle
(`config/ekf.yaml`) but is **not** used by the TF tree or the waypoint
follower. The map transform is owned entirely by `map_origin_publisher_node`,
which publishes `map` -> `base_footprint` at 10 Hz from GNSS/IMU plus a fixed
+90° CCW yaw offset (`YAW_OFFSET_RAD`).

## Documentation Index

| Doc | Contents |
|---|---|
| [docs/driver_station_waypoint_sim_testing.md](./docs/driver_station_waypoint_sim_testing.md) | Waypoint planning + laptop simulation design |
| [docs/waypoint_navigation_run_guide.md](./docs/waypoint_navigation_run_guide.md) | Live field-run guide (GNSS-only tracking) |
| [docs/wolf_encoder_odom.md](./docs/wolf_encoder_odom.md) | Encoder odometry package |
| [docs/Encoders.md](./docs/Encoders.md) | Encoder debugging log + wiring |
| [docs/jaus_interface.md](./docs/jaus_interface.md) | JAUS protocol notes |
| [docs/jaus_client.md](./docs/jaus_client.md) | JAUS Python client usage |
| [docs/wireless_config.md](./docs/wireless_config.md) | Static IP assignments |
| [docs/wolfpc_usb_paths.md](./docs/wolfpc_usb_paths.md) | Wolf PC USB device paths |
| [docs/VECTORNAV_ROS_TOPICS.md](./docs/VECTORNAV_ROS_TOPICS.md) | VectorNav topic reference |
| [docs/ethernet_to_dtm_connector.md](./docs/ethernet_to_dtm_connector.md) | Ethernet -> DTM connector how-to |
| [docs/Wolf IOP Ethernet Conversion.pdf](./docs/Wolf%20IOP%20Ethernet%20Conversion.pdf) | IOP -> panel Ethernet port conversion |
| [docs/files/H038455 - Wolf v2 IOP Interface Control Document_Rev01 (1).pdf](./docs/files/H038455%20-%20Wolf%20v2%20IOP%20Interface%20Control%20Document_Rev01%20(1).pdf) | Manufacturer IOP spec |
| [docs/files/Honeywell SNDH-T Series.pdf](./docs/files/Honeywell%20SNDH-T%20Series.pdf) | Wheel encoder datasheet |
| [docs/img/wolfrvizdemo.webm](./docs/img/wolfrvizdemo.webm) | Demo screen recording |

## Safety

- **Emergency Stop**: Connect the stock controller and use it as an E-STOP.
- **Auto-Release**: Control is automatically released on shutdown.
- **Motion stop**: Absence of an IOP command for >300 ms stops the vehicle.
- **Single `/joy` source**: `input_node` forces DISABLED whenever it sees an
  idle `/joy` (axes[5] = 0). Running two joystick sources (e.g. a host-side
  `joy_node` on the vehicle) would fight and the vehicle could not stay enabled.
- **Never run `wolf_fake_data` while linked to the vehicle**: a second
  `/nav/map_origin` + `/vectornav/gnss` corrupts the map anchor.

## Troubleshooting

### Cannot reach vehicle

- Verify network connectivity to vehicle (should be on same subnet)
- Ensure firewall allows UDP port 3794
- Check the static IPs in [docs/wireless_config.md](./docs/wireless_config.md)

### Control authority denied

- Vehicle may already be controlled by another node
- Check vehicle JAUS address configuration
- Verify Subsystem/Node/Component IDs match

### No joystick input

- Test with `ros2 topic echo /joy`
- Verify joystick is detected: `jstest /dev/input/js0`
- Check ROS2 domain ID matches (must be `42`)

### Map origin stuck at `inf` / `0,0`

- The origin only locks from a sane fix now, but the VectorNav reports `0,0`
  at boot before its INS converges. Wait for `/vectornav/gnss` to be sane, then
  restart `wolf_jaus`.

### Robot model heading off by 90 degrees

- Edit `YAW_OFFSET_RAD` in `map_origin_publisher_node.py` (+ = CCW, - = CW),
  then `docker compose restart wolf_jaus`.

### Topics don't bridge across the radio

- The radio link is IP (SSH works) but ROS 2 discovery is UDP multicast. If
  topics never appear on the DS, add a FastDDS discovery-server container on
  both sides (unicast works).

### Missing robot model on the DS

- The DS image must contain `wolf_description`. Rebuild:
  `docker compose build wolf_rviz`.

### AerialMap tile errors ("reset failed")

- Usually from a stale rviz_satellite cache/URL from a previous session. The
  current Esri URL serves tiles fine (live probe returns HTTP 200).

### Docker / compose gotchas

- `docker restart` preserves env; `docker compose up -d` re-reads it. After
  changing `MOTION` or other compose env, recreate with `up -d`, not `restart`.
- The PathManagerPanel `.so` is baked into the image by the Dockerfile
  `colcon build`. Only rebuild it if you edit `src/wolf_rviz_control/`.
- Python edits to `src/{path_manager,fake_data}` are hot (bind-mounted and
  shadow the image copies); edits apply on `restart` with no rebuild.
- Make sure `path_records/` exists at the repo root (the `/path_records` mount).
