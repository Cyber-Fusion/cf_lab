# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""cf_lab.learning — custom RL/distillation network architectures.

Importing this package has the side-effect of registering the custom
`StudentTeacherVision` class into the namespace of
`rsl_rl.runners.distillation_runner`, where the stock `DistillationRunner`
resolves `class_name` strings via `eval()`. The agent-cfg module
`tasks/manager_based/velocity/agents/rsl_rl_distillation_cfg.py` imports
this package precisely for that side-effect so the runner can find the
subclass when training the vision student.
"""

from cf_lab.learning.student_teacher_vision import StudentTeacherVision

# Inject into the runner's eval() lookup scope. The runner is what calls
# `eval(self.policy_cfg.pop("class_name"))`, so the name must be visible
# in *its* module globals.
import rsl_rl.runners.distillation_runner as _dr  # noqa: E402

_dr.StudentTeacherVision = StudentTeacherVision

__all__ = ["StudentTeacherVision"]
