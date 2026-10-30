#!/usr/bin/env bash
# Train the Kangaroo dance tracking experiments one after another.
#
#   D: range-aware action scale + strict lower body + action_rate curriculum
#   E: range-aware action scale + strict lower body + action_rate -0.1 + entropy 0.02
#   F: wide range-aware action scale + strict lower body + action_rate curriculum
#
# Usage (from anywhere):
#   scripts/train_dance_experiments.sh            # D, E and F
#   scripts/train_dance_experiments.sh D F        # only some of them
#   nohup scripts/train_dance_experiments.sh > /dev/null 2>&1 &   # survive logout
#
# Options via environment variables:
#   CSV=motions/<clip>.csv   motion to track (default: jazz clip, arms by position)
#   NUM_ENVS=4096            parallel environments
#   MAX_ITERS=1000           PPO iterations per experiment
#   LOGGER=tensorboard       tensorboard (default) or wandb
#   EXTRA_ARGS="..."         extra flags for every training, e.g.
#                            EXTRA_ARGS="--agent.actor.distribution-cfg.init-std 0.5"
#   DRY_RUN=1                print the commands without running them
#
# Everything about a session is saved in logs/dance_experiments/<timestamp>/:
#   session.log      full output of this script
#   config.txt       settings, git commit and GPU
#   git.diff         uncommitted changes at launch time
#   train_exp<X>.log training output of each experiment
#   runs.txt         experiment -> run directory (checkpoints, params, tfevents)
#   summary.csv      final metrics of each experiment
# Checkpoints and TensorBoard events stay in logs/rsl_rl/kangaroo_tracking/<run>.
# View them with: uv run tensorboard --logdir logs/rsl_rl/kangaroo_tracking
set -uo pipefail
cd "$(dirname "$0")/.."

CSV=${CSV:-motions/jazz_fivefoottwo_008_f400_armpos.csv}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERS=${MAX_ITERS:-1000}
LOGGER=${LOGGER:-tensorboard}
DRY_RUN=${DRY_RUN:-0}
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

NAME=$(basename "$CSV" .csv)
NPZ="$PWD/artifacts/$NAME/motion.npz"
RUNS_ROOT="logs/rsl_rl/kangaroo_tracking"
PREFIX="Mjlab-Tracking-Flat-Pal-Kangaroo"
declare -A TASKS=(
  [D]="$PREFIX-Range-Scale-Strict-Legs-AR-Curriculum"
  [E]="$PREFIX-Range-Scale-Strict-Legs-Entropy"
  [F]="$PREFIX-Wide-Range-Scale-Strict-Legs-AR-Curriculum"
)
EXPERIMENTS=("$@")
[ ${#EXPERIMENTS[@]} -eq 0 ] && EXPERIMENTS=(D E F)

die() { echo "ERROR: $*" >&2; exit 1; }
run() {
  echo "+ $*"
  [ "$DRY_RUN" = 1 ] || "$@"
}

# --- Preflight checks (fail now, not after hours of training) -----------------
for exp in "${EXPERIMENTS[@]}"; do
  [ -n "${TASKS[$exp]:-}" ] || die "unknown experiment '$exp' (valid: D E F)"
done
[ "$LOGGER" = tensorboard ] || [ "$LOGGER" = wandb ] || die "LOGGER must be tensorboard or wandb"
command -v uv > /dev/null || die "uv not found (https://docs.astral.sh/uv/)"
[ -f "$CSV" ] || [ -f "$NPZ" ] || die "motion not found: $CSV"
if [ "$DRY_RUN" != 1 ]; then
  nvidia-smi > /dev/null 2>&1 || die "no NVIDIA GPU available (nvidia-smi failed)"
  task_list=()
  for exp in "${EXPERIMENTS[@]}"; do task_list+=("${TASKS[$exp]}"); done
  uv run python - "${task_list[@]}" << 'PY' || die "tasks not registered (is this the right branch?)"
import sys
import pal_mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks
missing = [t for t in sys.argv[1:] if t not in list_tasks()]
if missing:
  sys.exit(f"missing tasks: {missing}")
PY
fi

# --- Session directory -----------------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
  SESSION=$(mktemp -d)
else
  SESSION="logs/dance_experiments/$(date '+%Y-%m-%d_%H-%M-%S')"
  mkdir -p "$SESSION"
fi
exec > >(tee -a "$SESSION/session.log") 2>&1
{
  echo "date:        $(date '+%F %T')"
  echo "experiments: ${EXPERIMENTS[*]}"
  for exp in "${EXPERIMENTS[@]}"; do echo "  $exp: ${TASKS[$exp]}"; done
  echo "csv:         $CSV"
  echo "npz:         $NPZ"
  echo "num_envs:    $NUM_ENVS"
  echo "max_iters:   $MAX_ITERS"
  echo "logger:      $LOGGER"
  echo "extra_args:  ${EXTRA_ARGS:-}"
  echo "git:         $(git rev-parse --abbrev-ref HEAD 2> /dev/null) $(git rev-parse HEAD 2> /dev/null)"
  echo "gpu:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2> /dev/null)"
} | tee "$SESSION/config.txt"
git diff > "$SESSION/git.diff" 2> /dev/null
echo "Session: $SESSION"
echo "TensorBoard: uv run tensorboard --logdir $RUNS_ROOT"

# --- Motion --------------------------------------------------------------------------
if [ ! -f "$NPZ" ]; then
  echo "== Converting $CSV -> $NPZ"
  run uv run python -m pal_mjlab.scripts.csv_to_npz --robot-name kangaroo \
    --input-file "$CSV" --output-name "$NAME" --input-fps 30 --output-fps 50 \
    || die "csv_to_npz failed"
fi

# --- Training ------------------------------------------------------------------------
echo "experiment,task,status,run_dir" > "$SESSION/runs.txt"
declare -A STATUS RUN_DIR
for exp in "${EXPERIMENTS[@]}"; do
  run_name="exp$exp-$NAME"
  echo "== Experiment $exp: ${TASKS[$exp]} ($(date '+%F %T'))"
  # "uv run python -m" uses this project's venv even if its console scripts
  # (.venv/bin/train) point to another checkout.
  cmd=(uv run python -m mjlab.scripts.train "${TASKS[$exp]}"
    --env.commands.motion.motion-file "$NPZ"
    --env.scene.num-envs "$NUM_ENVS"
    --agent.max-iterations "$MAX_ITERS"
    --agent.logger "$LOGGER"
    --agent.run-name "$run_name"
    "${EXTRA[@]}")
  echo "+ ${cmd[*]}"
  code=0
  [ "$DRY_RUN" = 1 ] || "${cmd[@]}" > "$SESSION/train_exp$exp.log" 2>&1 || code=$?
  STATUS[$exp]=$([ $code -eq 0 ] && echo ok || echo "failed($code)")
  RUN_DIR[$exp]=$(ls -td "$RUNS_ROOT"/*_"$run_name" 2> /dev/null | head -1)
  echo "$exp,${TASKS[$exp]},${STATUS[$exp]},${RUN_DIR[$exp]}" >> "$SESSION/runs.txt"
  echo "   -> ${STATUS[$exp]}  ${RUN_DIR[$exp]:-<no run dir>}"
  [ $code -eq 0 ] || tail -n 20 "$SESSION/train_exp$exp.log"
done

# --- Summary -------------------------------------------------------------------------
if [ "$DRY_RUN" != 1 ]; then
  uv run python - "$SESSION" "${EXPERIMENTS[@]}" << 'PY'
import csv
import re
import sys
from pathlib import Path

session, experiments = Path(sys.argv[1]), sys.argv[2:]
metrics = {
  "iteration": r"Learning iteration (\d+)/",
  "mean_reward": r"Mean reward: ([-\d.]+)",
  "episode_length": r"Mean episode length: ([-\d.]+)",
  "joint_pos_error": r"error_joint_pos: ([-\d.]+)",
  "body_rot_error": r"error_body_rot: ([-\d.]+)",
  "body_pos_error": r"error_body_pos: ([-\d.]+)",
  "body_ori_reward": r"Episode_Reward/motion_body_ori: ([-\d.]+)",
  "legs_pos_reward": r"Episode_Reward/motion_lower_body_pos: ([-\d.]+)",
  "legs_ori_reward": r"Episode_Reward/motion_lower_body_ori: ([-\d.]+)",
  "legs_vel_reward": r"Episode_Reward/motion_lower_body_lin_vel: ([-\d.]+)",
  "action_rate": r"Episode_Reward/action_rate_l2: ([-\d.]+)",
  "action_std": r"Mean action std: ([-\d.]+)",
}
rows = []
for exp in experiments:
  log = session / f"train_exp{exp}.log"
  text = log.read_text(errors="ignore") if log.exists() else ""
  last = text.split("Learning iteration ")[-1] if "Learning iteration " in text else ""
  row = {"experiment": exp}
  for key, pattern in metrics.items():
    source = "Learning iteration " + last if key == "iteration" else last
    match = re.search(pattern, source)
    row[key] = match.group(1) if match else ""
  rows.append(row)
with open(session / "summary.csv", "w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=["experiment", *metrics])
  writer.writeheader()
  writer.writerows(rows)
print("== Final metrics")
for row in rows:
  print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
PY
fi

echo "== Summary ($(date '+%F %T'))"
for exp in "${EXPERIMENTS[@]}"; do
  echo "  $exp ${TASKS[$exp]}: ${STATUS[$exp]}  ${RUN_DIR[$exp]:-}"
done
echo "Session data: $SESSION"
echo "TensorBoard:  uv run tensorboard --logdir $RUNS_ROOT"
