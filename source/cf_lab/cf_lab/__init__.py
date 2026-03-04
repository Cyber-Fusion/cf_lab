# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os

CF_LAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Absolute path to the extension source directory."""

CF_LAB_DATA_DIR = os.path.join(CF_LAB_EXT_DIR, "data")
"""Absolute path to the extension data directory."""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
