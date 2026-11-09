#!/bin/bash
# RunPod entrypoint: SSH, a pod-lifetime budget, one-time data setup, then training + evaluation.
# Env (set in the RunPod template): RUN (lafan1_main), ITERS (16000), BUDGET_HOURS (3.5),
# AUTOSTOP (1: stop the pod after training/eval or when the budget is spent), TRAIN_ARGS, EVAL_ARGS,
# MOTIONS (lafan1 | bones: BONES-SEED, needs HF_TOKEN of an account that accepted its license),
# SKIP_GPU_CHECK (1: let the preflight pass without a GPU, for local tests only).
set -uo pipefail
RUN=${RUN:-lafan1_main}
ITERS=${ITERS:-16000}
BUDGET_HOURS=${BUDGET_HOURS:-3.5}
AUTOSTOP=${AUTOSTOP:-1}
MOTIONS=${MOTIONS:-lafan1}
VOL=/workspace/supermimic
cd /opt/supermimic
BONES=data/bones_seed
BONES_NPZ=$BONES/g1_50hz_noprops.npz
BONES_CLIPS=130304  # motions left by tools/pack_motions.py --drop_support_props
MOTION_ARG=""
[ "$MOTIONS" = bones ] && MOTION_ARG="--motion_dir $BONES_NPZ"

stop_pod() {
  echo "[start.sh] $(date -u +%FT%TZ) stopping pod: $1" | tee -a "$VOL/start.log"
  [ "$AUTOSTOP" = 1 ] || return 0
  runpodctl pod stop "${RUNPOD_POD_ID:-}" || echo "[start.sh] runpodctl stop FAILED; stop the pod by hand" | tee -a "$VOL/start.log"
}

# training, then evaluation of any checkpoints the watcher has not scored, then stop (runs in tmux)
if [ "${1:-}" = pipeline ]; then
  RESUME=""
  [ -f "logs/$RUN/last.pt" ] && RESUME="--resume logs/$RUN/last.pt"
  python supermimic/train.py --log_dir "logs/$RUN" --iterations "$ITERS" $RESUME $MOTION_ARG ${TRAIN_ARGS:-} 2>&1 | tee -a "logs/$RUN/stdout.log"
  rc=${PIPESTATUS[0]}
  tmux kill-session -t eval_main
  python supermimic/evaluate.py --run_dir "logs/$RUN" $MOTION_ARG ${EVAL_ARGS:-} 2>&1 | tee -a "logs/$RUN/eval_stdout.log"
  if [ "$rc" -ne 0 ]; then stop_pod "training FAILED (exit $rc), see logs/$RUN/stdout.log"; else stop_pod "training and evaluation finished"; fi
  exit 0
fi

# SSH with the public key RunPod injects
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p ~/.ssh /run/sshd && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
  ssh-keygen -A && /usr/sbin/sshd
fi

# persistent state on the volume
mkdir -p "$VOL/data" "$VOL/logs" "$VOL/.cache"
for d in data logs .cache; do ln -sfn "$VOL/$d" "/opt/supermimic/$d"; done
[ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] || echo "[start.sh] WARNING: RUNPOD_API_KEY/RUNPOD_POD_ID missing; auto-stop will not work" | tee -a "$VOL/start.log"

# budget is running time summed over all boots of this volume (.used_seconds, ticked every 30 s), so a restart
# continues it rather than resetting it, and time spent stopped does not count
BUDGET_S=$(python3 -c "print(int($BUDGET_HOURS * 3600))")
USED=$(cat "$VOL/.used_seconds" 2>/dev/null || echo 0)
echo "[start.sh] $(date -u +%FT%TZ) boot; budget left $((BUDGET_S - USED))s" | tee -a "$VOL/start.log"
if [ "$USED" -ge "$BUDGET_S" ]; then stop_pod "budget spent"; exec sleep infinity; fi
(while sleep 30; do
  USED=$((USED + 30)) && echo "$USED" > "$VOL/.used_seconds"
  if [ "$USED" -ge "$BUDGET_S" ]; then stop_pod "budget of ${BUDGET_HOURS} h reached"; exit; fi
done) &

# preflight: fail fast, before any long step, if the environment is broken
if ! python -c "import isaaclab, isaaclab_newton, torch; assert torch.cuda.is_available() or '${SKIP_GPU_CHECK:-0}' == '1', 'no CUDA GPU visible'" >> "$VOL/start.log" 2>&1; then
  stop_pod "preflight FAILED (Isaac Lab import or GPU), see start.log"; exec sleep infinity
fi
echo "[start.sh] preflight ok" | tee -a "$VOL/start.log"

# one-time data download + MJCF/USD conversion, retried and verified (40 clips, 38 meshes, MJCF, USD)
data_ok() {
  [ "$(ls data/lafan1_g1/*.csv 2>/dev/null | wc -l)" -eq 40 ] && [ "$(ls data/g1_description/meshes/*.STL 2>/dev/null | wc -l)" -eq 38 ] \
    && [ -f data/g1_description/g1_29dof.xml ] && [ -f data/g1_usd/g1_29dof/g1_29dof.usda ]
}
for attempt in 1 2 3; do
  data_ok && break
  echo "[start.sh] data setup attempt $attempt" | tee -a "$VOL/start.log"
  tools/download_data.sh >> "$VOL/start.log" 2>&1
done
if ! data_ok; then stop_pod "data setup FAILED after 3 attempts, see start.log"; exec sleep infinity; fi
echo "[start.sh] data ok" | tee -a "$VOL/start.log"

# BONES-SEED: stream the G1 archive (23.5 GB, 142k CSVs) straight into the 50 Hz packer and verify the result. The
# CSVs are never extracted: writing that many small files to the network volume is slow (and tar cannot chown there).
bones_ok() {
  [ -f "$BONES_NPZ" ] && [ "$(python -c "import numpy as np; print(len(np.load('$BONES_NPZ')['lens']))" 2>/dev/null)" = "$BONES_CLIPS" ]
}
bones_setup() {
  local HFD=https://huggingface.co/datasets/bones-studio/seed/resolve/main
  mkdir -p "$BONES/metadata"
  curl -sfL --retry 3 -H "Authorization: Bearer $HF_TOKEN" -o "$BONES/metadata/seed_metadata_v004.parquet" "$HFD/metadata/seed_metadata_v004.parquet" || return 1
  python -c "import pandas, pyarrow" 2>/dev/null || pip install -q pandas pyarrow || return 1
  curl -sfL --retry 3 -H "Authorization: Bearer $HF_TOKEN" "$HFD/g1.tar.gz" \
    | python tools/pack_motions.py - "$BONES_NPZ" --metadata "$BONES/metadata/seed_metadata_v004.parquet" --drop_support_props
}
if [ "$MOTIONS" = bones ]; then
  [ -d "$BONES/g1" ] && (rm -rf "$BONES/g1" &)  # CSVs left by the earlier extract-to-disk setup
  for attempt in 1 2 3; do
    bones_ok && break
    [ -n "${HF_TOKEN:-}" ] || break
    echo "[start.sh] $(date -u +%FT%TZ) BONES-SEED setup attempt $attempt" | tee -a "$VOL/start.log"
    bones_setup 2>&1 | tee -a "$VOL/start.log"
  done
  if ! bones_ok; then stop_pod "BONES-SEED setup FAILED (HF_TOKEN set: ${HF_TOKEN:+yes}), see start.log"; exec sleep infinity; fi
  echo "[start.sh] $(date -u +%FT%TZ) BONES-SEED ok" | tee -a "$VOL/start.log"
fi

mkdir -p "logs/$RUN"
tmux new-session -d -s eval_main "python supermimic/evaluate.py --run_dir logs/$RUN --watch $MOTION_ARG ${EVAL_ARGS:-} 2>&1 | tee -a logs/$RUN/eval_stdout.log"
tmux new-session -d -s "$RUN" "bash /opt/supermimic/docker/start.sh pipeline"
exec sleep infinity
