from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from pal_mjlab.robots import (
  KANGAROO_RANGE_ACTION_SCALE,
  KANGAROO_WIDE_RANGE_ACTION_SCALE,
)

from .env_cfgs import (
  pal_kangaroo_flat_tracking_env_cfg,
  pal_kangaroo_hands_flat_env_cfg,
)
from .rl_cfg import pal_kangaroo_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(play=True),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Action scale 0.25 * max(effort / Kp, half joint range), stricter tracking of
# the lower body and an action_rate curriculum (-0.01 -> -0.1).
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-Range-Scale-Strict-Legs-AR-Curriculum",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    action_scale=KANGAROO_RANGE_ACTION_SCALE,
    strict_lower_body=True,
    action_rate_curriculum=True,
  ),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    play=True, action_scale=KANGAROO_RANGE_ACTION_SCALE, strict_lower_body=True
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Same as above without curriculum: fixed action_rate -0.1, and a higher PPO
# entropy coefficient (0.005 -> 0.02) to keep the exploration noise from
# collapsing.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-Range-Scale-Strict-Legs-Entropy",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    action_scale=KANGAROO_RANGE_ACTION_SCALE, strict_lower_body=True
  ),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    play=True, action_scale=KANGAROO_RANGE_ACTION_SCALE, strict_lower_body=True
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(entropy_coef=0.02),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Same as the curriculum task with a wider action scale (range term 0.35
# instead of 0.25): arms explore 1.4x more, legs are essentially unchanged.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-Wide-Range-Scale-Strict-Legs-AR-Curriculum",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    action_scale=KANGAROO_WIDE_RANGE_ACTION_SCALE,
    strict_lower_body=True,
    action_rate_curriculum=True,
  ),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    play=True, action_scale=KANGAROO_WIDE_RANGE_ACTION_SCALE, strict_lower_body=True
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-No-State-Estimation",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    has_state_estimation=False, play=True
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-Hands",
  env_cfg=pal_kangaroo_hands_flat_env_cfg(has_state_estimation=True),
  play_env_cfg=pal_kangaroo_hands_flat_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-Hands-No-State-Estimation",
  env_cfg=pal_kangaroo_hands_flat_env_cfg(has_state_estimation=False),
  play_env_cfg=pal_kangaroo_hands_flat_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
