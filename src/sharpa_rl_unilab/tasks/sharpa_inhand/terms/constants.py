"""Immutable Sharpa task contracts used by Manager-Based terms."""

from __future__ import annotations

import numpy as np

NUM_HAND_JOINTS = 22
HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)
ACTUATOR_NAMES: tuple[str, ...] = tuple(f"{name}_ctrl" for name in HAND_JOINT_NAMES)
TACTILE_SENSOR_NAMES: tuple[str, ...] = (
    "contact_right_thumb_elastomer_force",
    "contact_right_index_elastomer_force",
    "contact_right_middle_elastomer_force",
    "contact_right_ring_elastomer_force",
    "contact_right_pinky_elastomer_force",
)
SOURCE_DEFAULT_HAND_JOINT_POS_DEG: tuple[float, ...] = (
    95.12771,
    -3.11244,
    14.81626,
    -1.03493,
    12.23986,
    65.21091,
    6.1133,
    15.58495,
    5.90325,
    31.74149,
    -0.95812,
    41.88173,
    12.844,
    31.72383,
    9.84458,
    35.22366,
    18.02839,
    10.9712,
    68.30895,
    7.99151,
    5.89626,
    5.89875,
)
SOURCE_DEFAULT_HAND_JOINT_POS = np.deg2rad(
    np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float64)
)
