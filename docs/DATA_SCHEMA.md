# GrandTour episode data schema (Option B / guide-faithful)

The loader contract inside jepa-wms (`DROIDVideoDataset` franka-custom branch,
`app/plan_common/datasets/droid_dset.py`) dictates the *item* format a dataset
class must return. Because guide-faithful actions (commanded twist) and state
(joint encoders) cannot be expressed in that loader's 7-dim pose format, this
branch adds a purpose-built class + on-disk schema.

## Layout

    $JEPAWM_DSET/grand_tour/data/<mission>/episode.h5

one h5 per mission (mirrors `franka_custom/data/<task>/<run>/episode.h5`).
A mission is a short (~3-8 min) recording; episodes are consumed by
`GrandTourVideoDataset`, which samples random windows of `frames_per_clip`
model frames at stride computed from the true camera rate.

## H5 contents

| entry | dtype / shape | meaning |
|---|---|---|
| attr `camera` | str | "hdr_front" |
| attr `n_frames` | int | number of camera frames |
| `frames_bytes` | uint8 [M] | concatenated JPEGs, resized to short side 512 |
| `frames_offsets` | int64 [N+1] | byte offsets; `[i], [i+1]` = frame i (offset[0] = 0) |
| `cam_timestamp` | float64 [N] | camera timestamps (seconds) |
| `cmd_twist` | float32 [N,3] | commanded twist `[vx, vy, yaw_rate]` |
| `joints` | float32 [N,36] | 12 joints x [position, velocity, torque] |
| `pose` | float32 [N,7] | base pose `[x,y,z, qx,qy,qz,qw]` (dlio odometry) |

All non-frame arrays are aligned to camera frame times by nearest-timestamp
match at build time (`scripts/build_episodes.py`).

## Sensor facts verified on the data (2024-10/11 ETH missions)

- `images/hdr_front` + `data/hdr_front`: **10 fps** jpegs with timestamps;
  file index == timestamp index.
- `data/anymal_command_twist`: commanded body twist, ~16-19 Hz, keys
  `{timestamp, linear [3], angular [3]}`. Commanded signal exists (not just
  realized motion); `[vx, vy, yaw]` used as the action.
- `data/anymal_state_actuator`: 400 Hz, per-joint (`00`..`11`) fields
  `{*_state_joint_position, *_state_joint_velocity, *_state_joint_torque, ...}`.
- `data/dlio_map_odometry`: 10 Hz base pose `{pose_pos [3], pose_orien [4] quat}`.
- Timestamps are stored in **seconds** (float64).

## Dataset class semantics (app/plan_common/datasets/grandtour_dset.py)

`GrandTourVideoDataset.__getitem__` returns the same 4-tuple as the
franka-custom loader:
`(obs, actions, states, reward)` with
`obs = {"visual": TCHW float [0,1] pre-norm, "proprio": joints normalized [T,36]}`,
`actions = concat([cmd_twist, (joints-mean)/std])` shape `[T,39]`
(guide section 6: "3 or 3+36 if concatenating state"), `states = pose [T,7]`.
Per-dataset joint mean/std are computed over sampled rows at init; cmd/pose are
kept raw (identity stats).
