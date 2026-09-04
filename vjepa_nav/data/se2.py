"""SE(2) helpers ported from LiMo's dataset_builder (BSD-3)."""

from __future__ import annotations

import numpy as np


def convert_se2_to_transform(se2: np.ndarray) -> np.ndarray:
    se2 = np.asarray(se2, dtype=float)
    x, y, yaw = se2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


def transform_se2_odom_to_base(
    se2_points_odom: np.ndarray, T_odom_base: np.ndarray
) -> np.ndarray:
    """Map SE(2) poses from the odom (world) frame into the robot base frame.

    T_odom_base is the 4x4 homogeneous transform of the robot base in the
    odom/world frame at the reference time.
    """
    T = np.asarray(T_odom_base, dtype=float)
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    r_base_odom = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    ).T
    t_xy = T[:2, 3:4]
    t_base_odom = -r_base_odom @ t_xy

    xy = np.asarray(se2_points_odom, dtype=float).T[:2]
    xy_base = r_base_odom @ xy + t_base_odom
    yaw_base = np.asarray(se2_points_odom, dtype=float).T[2] - yaw
    yaw_base = np.arctan2(np.sin(yaw_base), np.cos(yaw_base))
    return np.column_stack([xy_base.T, yaw_base])
