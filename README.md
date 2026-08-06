# Wolf ROS2 JAUS Bridge

ROS2 interface for the Wolf DBW vehicle with JAUS protocol support.

## Overview

This package provides a ROS2 bridge to the Wolf DBW vehicle's JAUS interface, allowing control via joystick and publishing vehicle state to standard ROS2 topics.

It also supports **waypoint navigation**: waypoints are clicked on a live satellite map in RViz on the driver station, sent to the vehicle, and tracked using the VectorNav GNSS/IMU. A `fake_data` package simulates the whole vehicle on a laptop so the full flow can be tested without hardware.

## Architecture

```
┌─────────────────┐     Silvus Radio       ┌─────────────────┐     Ethernet           ┌─────────────────┐
│  DS PI Nodes    │◄─(192.168.0.156)       │  WOLF PC Nodes  │◄──────────────────────►│ Wolf DBW Vehicle│
│ (192.168.0.52)  │(192.168.0.159)─►       │ (192.168.0.50)  │                        │ (192.168.1.42)  │
│  - joy_node     │                        │ (192.168.1.50)  │                        │  PrimitiveDriver│
│  - path_manager │                        │  - drive_node   │                        │  VelocitySensor │
│  - rosboard     │                        │  - input_node   │                        └─────────────────┘
└─────────────────┘                        │  - pathing_node │
         │                                 │  - teleop_node  │
         │                                 │  - path_recorder│
         ▼                                 │  - map_origin   │
       /joy                                │  - vectornav    │
                                           └─────────────────┘
                                                    │
                                                    ▼
                                                /cmd_vel
                                             /cmd_vel_teleop
                                            /cmd_vel_pathing
                                              /input/state
                                               /input/mode
                                         request_state_change
```

## Quick Start

### Clone the repo on both PIs

```bash
git clone https://github.com/tamu-edu/wolf_ros.git --recursive
```

> `src/vectornav` is a **git submodule** (private repo). The DS image never copies it, so only the vehicle clone needs `--recursive`. The `tamu-edu/wolf_ros` repo itself is **private** - a plain `https://` clone prompts for a GitHub username/password and fails non-interactively. Use an SSH clone (`git clone --recurse-submodules git@github.com:tamu-edu/wolf_ros.git`) or a personal access token with https.

### Configure network settings

Configure network settings as noted [here](./docs/wireless_config.md)

### Run start scripts

- `/scripts/run_wolf.sh` - Runs on the wolf pc (adds the 192.168.1.50 secondary IP and brings up the vehicle containers)
- `/scripts/add-ip.sh` - Runs on the wolf pc to add second ip address
- `/scripts/run_ds.sh` - Runs on the driverstation pi (fixes the Xauthority file so the RViz window appears on the Pi desktop, then brings up the driver station containers)

### Build and run the containers

Driver station (DS Pi):

```bash
cd docker_ds
docker compose build wolf_rviz        # only needed after editing src/ packages
docker compose up -d wolf_ds wolf_rviz
```

Vehicle (Wolf PC):

```bash
cd docker_wolf
docker compose build wolf_jaus wolf_localization wolf_description
docker compose up -d
```

Laptop simulation (fake vehicle, no hardware):

```bash
cd docker_ds
docker compose build
docker compose -f docker-compose.yml \
               -f docker-compose.laptop.yml \
               -f docker-compose.fake.yml up -d \
               wolf_ds wolf_fake_data wolf_rviz
```

The `wolf_jaus` container bind-mounts `src/wolf_jaus_bridge`, so Python source edits are live after `docker compose restart wolf_jaus` - no rebuild. The launch file and installed executables come from the image build, so rebuild after changing those.

## ROS2 Topics

### Command Topics

- `/joy` (sensor_msgs/msg/Joy) - Raw joystick input from the driver station `joy_node`. This is the only `/joy` source on the network.
- `/cmd_vel` (geometry_msgs/msg/Twist) - Final velocity commands (input_node → drive_node) that are sent to the vehicle over JAUS.
- `/cmd_vel_teleop` (geometry_msgs/msg/Twist) - Joystick-derived velocity commands (teleop_node → input_node). Only forwarded to `/cmd_vel` while in TELEOP mode.
- `/cmd_vel_pathing` (geometry_msgs/msg/Twist) - Autonomous steering commands (pathing_node → input_node). Only forwarded to `/cmd_vel` while in PATHING mode.

### Input Topics

- `/input/mode` (std_msgs/msg/String) - Current drive mode (`TELEOP` or `PATHING`), published by input_node.
- `/input/state` (std_msgs/msg/String) - Current control state (`DISABLED` or `ENABLED`), published by input_node.

### Navigation Topics

- `/nav/map_origin` (geometry_msgs/msg/PointStamped) - The fixed UTM map origin (easting/northing in point.x/y), published at 1 Hz by the vehicle's `map_origin_publisher_node` (or `fake_data_node` on a laptop). Defines the `map` frame.
- `/nav/map_origin_fix` (sensor_msgs/msg/NavSatFix) - The map origin converted back to a constant lat/lon fix by path_manager_node. The RViz AerialMap anchors to this so the satellite image stays still while the vehicle drives.
- `/nav/waypoint_path` (nav_msgs/msg/Path) - The waypoint path published by path_manager_node. Lon is stored in pose.position.x, lat in pose.position.y, frame `gps`. The follower tracks this.
- `/nav/path_manager/preview` (nav_msgs/msg/Path) - The current waypoints in map-frame XY, published so RViz can display them as markers.
- `/nav/path_manager/send` (std_msgs/msg/Empty) - Sent by the RViz PathManagerPanel to send the clicked waypoints as the active path.
- `/nav/path_manager/undo` (std_msgs/msg/Empty) - Sent by the panel to remove the last clicked waypoint.
- `/nav/path_manager/clear` (std_msgs/msg/Empty) - Sent by the panel to clear all waypoints; path_manager also publishes an empty path so the vehicle stops.
- `/nav/path_manager/save` (std_msgs/msg/Empty) - Sent by the panel to save the current path.
- `/nav/path_manager/save_trigger` (std_msgs/msg/Empty) - Published by path_manager_node to tell the vehicle's `path_recorder_node` to write the path to file.
- `/nav/path_manager/status` (std_msgs/msg/String) - Status text from path_manager_node shown in the panel status box.
- `/nav/path_status` (std_msgs/msg/String) - Follower progress from `pathing_node` (or `fake_data_node`): "Received a new path...", "Waypoint N/M reached!", "Path finished".
- `/clicked_point` (geometry_msgs/msg/PointStamped) - Output of the RViz Publish Point tool (frame `map`). Each click becomes a waypoint.
- `nav/dist_error` (std_msgs/msg/Float32) - Distance from the vehicle to the current waypoint (pathing_node).
- `nav/heading_error` (std_msgs/msg/Float32) - Heading error to the current waypoint (pathing_node).

### VectorNav Topics

Full field-by-field reference: [docs/VECTORNAV_ROS_TOPICS.md](./docs/VECTORNAV_ROS_TOPICS.md)

- `/vectornav/gnss` (sensor_msgs/msg/NavSatFix) - GNSS fix (lat/lon/alt). The waypoint follower tracks this for position.
- `/vectornav/imu` (sensor_msgs/msg/Imu) - Calibrated IMU (orientation, angular velocity, linear acceleration). The follower uses this for heading.
- `/vectornav/imu_uncompensated` (sensor_msgs/msg/Imu) - Raw, uncalibrated IMU data.
- `/vectornav/magnetic` (sensor_msgs/msg/MagneticField) - Magnetometer reading in Tesla.
- `/vectornav/pose` (geometry_msgs/msg/PoseWithCovarianceStamped) - Filtered pose estimate with covariance from the VectorNav INS.
- `/vectornav/pressure` (sensor_msgs/msg/FluidPressure) - Barometric pressure reading.
- `/vectornav/raw/attitude` (vectornav_msgs/msg/AttitudeGroup) - Raw attitude group (VPE status, yaw/pitch/roll, quaternion, DCM, etc.).
- `/vectornav/raw/common` (vectornav_msgs/msg/CommonGroup) - Raw common group (status/time registers).
- `/vectornav/raw/gps` (vectornav_msgs/msg/GpsGroup) - Raw GPS composite data (UTC, fix type, lat/lon/height, ECEF, velocities, DOP).
- `/vectornav/raw/gps2` (vectornav_msgs/msg/GpsGroup) - Raw data from the second GPS receiver.
- `/vectornav/raw/imu` (vectornav_msgs/msg/ImuGroup) - Raw IMU composite group.
- `/vectornav/raw/ins` (vectornav_msgs/msg/InsGroup) - Raw INS composite group (position/velocity/attitude solutions).
- `/vectornav/raw/time` (vectornav_msgs/msg/TimeGroup) - Raw time group (startup/GPS/UTC time, sync-in/out counters).
- `/vectornav/temperature` (sensor_msgs/msg/Temperature) - Sensor temperature in Celsius.
- `/vectornav/time_gps` (sensor_msgs/msg/TimeReference) - GPS time reference.
- `/vectornav/time_pps` (sensor_msgs/msg/TimeReference) - Pulse-per-second time reference.
- `/vectornav/time_startup` (sensor_msgs/msg/TimeReference) - Time since sensor startup.
- `/vectornav/time_syncin` (sensor_msgs/msg/TimeReference) - External sync-in time reference.
- `/vectornav/velocity_aiding` (geometry_msgs/msg/Twist) - Velocity aiding data fed into the INS filter.
- `/vectornav/velocity_body` (geometry_msgs/msg/TwistWithCovarianceStamped) - Body-frame velocity estimate with covariance.

### Other Topics

- `/odom` (nav_msgs/msg/Odometry) - Wheel encoder odometry from `wolf_encoder_odom`. Not used in the current GNSS-only setup.
- `/robot_description` (std_msgs/msg/String) - URDF robot description, published by `wolf_description` (robot_state_publisher).
- `/tf` / `/tf_static` (tf2_msgs/msg/TFMessage) - The transform tree: `map` → `base_footprint` (published by map_origin_publisher_node) and the URDF chain down to the wheel links.

### Services

- `request_state_change` (jaus_interface/srv/RequestStateChange) - Requests or disengages control of the vehicle. `input_node` calls it when the deadman switch (RT) is pressed/released. Togglable.

## Joystick Configuration

Default mapping (PS4/Xbox layout):
- **Left Stick Vertical (Axis 1)**: Forward/Reverse velocity
- **Right Stick Horizontal (Axis 3)**: Left/Right rotation
- **Right Trigger (Axis 5)**: Deadman switch / enable (hold)
- **A (Button 0)**: Teleop mode toggle
- **B (Button 1)**: Pathing mode toggle
- **X (Button 2)**: Clear the path (`path_recorder_node`)
- **Y (Button 3)**: Add the current GPS waypoint to the path (`path_recorder_node`)
- **LB (Button 4)**: Cycle / load a saved path file (`path_recorder_node`)
- **RB (Button 5)**: Save the path (`path_recorder_node`)

Button/axis indices are standard Linux gamepad indices - confirm with `ros2 topic echo /joy` on the DS Pi. The vehicle only moves when mode is `PATHING` **and** state is `ENABLED` (RT held); in `TELEOP` mode it follows the sticks.

## JAUS Information

- JAUS Protocol: Read more [here](/docs/jaus_interface.md)
- JAUS Interface: Read more [here](/docs/jaus_client.md)

Key facts:

- The vehicle runs JAUS at `192.168.1.42:3794`. The client uses its own JAUS address (e.g. `1.100.1`) and talks to the vehicle component `1.2.1`.
- Control must be re-requested at roughly 3 Hz (control times out after 0.5 s). Drive with `set_wrench_effort()` at roughly 20 Hz for responsiveness.
- Wrench mapping: ±100% = ±6.26 m/s linear / ±4.71 rad/s rotational. Internally this maps to motor RPM with track width 1.30 m, wheel radius 0.33 m, and gear ratio 30.0.
- The Wolf only advertises 9 JAUS services. `PrimitiveDriver` and `PowerPlantManager` are hidden from `DiscoveryService` but still work when you send to them directly. Battery/genset data is not available over JAUS.
- Absence of an IOP command for >300 ms triggers a motion stop. The stock controller's E-STOP always works.

## Safety

- **Emergency Stop**: Connect the stock controller and use it as an E-STOP.
- **Auto-Release**: Control is automatically released on shutdown.
- **Motion stop**: Absence of an IOP command for >300 ms stops the vehicle.
- **Single joystick source**: Only the DS Pi publishes `/joy`. `input_node` forces DISABLED whenever it sees an idle `/joy` (axes[5] = 0), so a second source (e.g. a host-side `joy_node` on the vehicle) would fight and the vehicle could not stay enabled.
- **Simulation isolation**: Never run `wolf_fake_data` while linked to the vehicle - a second `/nav/map_origin` and `/vectornav/gnss` corrupts the map anchor.

## Troubleshooting

### Cannot reach vehicle

- Verify network connectivity to vehicle (should be on same subnet)
- Ensure firewall allows UDP port 3794

### Control authority denied

- Vehicle may already be controlled by another node
- Check vehicle JAUS address configuration
- Verify Subsystem/Node/Component IDs match

### No joystick input

- Test with `ros2 topic echo /joy`
- Verify joystick is detected: `jstest /dev/input/js0`
- Check ROS2 domain ID matches

### Map origin stuck at `inf` / `0,0`

- The origin only locks from a sane fix now, but the VectorNav reports `0,0` at boot before its INS converges. Wait for `/vectornav/gnss` to be sane, then restart `wolf_jaus`.

### Robot model heading off by 90 degrees

- Edit `YAW_OFFSET_RAD` in `map_origin_publisher_node.py` (+ = CCW, - = CW), then `docker compose restart wolf_jaus`.

### Topics don't bridge across the radio

- The radio link is IP (SSH works) but ROS 2 discovery is UDP multicast. If topics never appear on the DS, add a FastDDS discovery-server container on both sides (unicast works).

### Missing robot model on the DS

- The DS image must contain `wolf_description`. Rebuild: `docker compose build wolf_rviz`.

### AerialMap tile errors ("reset failed")

- Usually from a stale rviz_satellite cache/URL from a previous session. The current Esri URL serves tiles fine.

### Docker gotchas

- `docker restart` preserves env; `docker compose up -d` re-reads it. After changing `MOTION` or other compose env, recreate with `up -d`, not `restart`.
- The PathManagerPanel `.so` is baked into the image by the Dockerfile `colcon build`. Only rebuild it if you edit `src/wolf_rviz_control/`.
- Python edits to `src/{path_manager,fake_data}` are hot (bind-mounted and shadow the image copies); they apply on `restart` with no rebuild.
- Make sure `path_records/` exists at the repo root (the `/path_records` mount).
