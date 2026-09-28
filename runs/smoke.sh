#!/bin/bash

# Tiny end-to-end check of both arms on the real packed data, meant to take minutes: a few pretraining steps
# with periodic val evals and a mid-run checkpoint, then the second half again resumed from that checkpoint
# (should end at about the same val bpb, since the packed loader resumes exactly), then base_eval.
# Run it after runs/data.sh and before anything long.
#
#   bash runs/smoke.sh
#   SMOKE_DEPTH=20 bash runs/smoke.sh                        # does the default device batch size fit at d20?
#   SMOKE_DEPTH=20 DEVICE_BATCH_SIZE=4 bash runs/smoke.sh    # ... or does 4?
#
# Checkpoints: base_checkpoints/smoke_<arm>_d<depth>, logs: $NANOCHAT_BASE_DIR/experiments/smoke/

source "$(dirname "$0")/common.sh"

DEPTH="${SMOKE_DEPTH:-4}"
STEPS="${SMOKE_STEPS:-20}"
ARMS=(${ARMS:-baseline cheat_sheet})
OUT_DIR="$EXPERIMENTS_DIR/smoke"
mkdir -p "$OUT_DIR"

require_data
require_free_gpus

for arm in "${ARMS[@]}"; do
    check_arm "$arm"
    TAG="smoke_${arm}_d${DEPTH}"
    DBS=$(device_batch_size "$arm" "$DEPTH")
    LOG="$OUT_DIR/${TAG}.log"
    : > "$LOG"
    log "Smoke test: $arm d$DEPTH, device batch size $DBS (log: $LOG)"

    TRAIN_ARGS=(
        "${COMMON_TRAIN_ARGS[@]}" $(arm_args "$arm")
        --depth="$DEPTH"
        --device-batch-size="$DBS"
        --num-iterations="$STEPS"
        --eval-every=$((STEPS / 2))
        --eval-tokens=$((8 * 2048 * NPROC)) # 8 rows per GPU
        --save-every=$((STEPS / 2))
        --model-tag="$TAG"
        --run=dummy
    )
    run_logged "$LOG" launch scripts.base_train -- "${TRAIN_ARGS[@]}"
    FIRST_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')

    log "Resuming $TAG from step $((STEPS / 2))"
    run_logged "$LOG" launch scripts.base_train -- "${TRAIN_ARGS[@]}" --resume-from-step=$((STEPS / 2))
    RESUMED_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')

    run_logged "$LOG" launch scripts.base_eval -- --model-tag="$TAG" --eval=bpb --device-batch-size="$DBS" --split-tokens=$((8 * 2048 * NPROC))

    log "$arm d$DEPTH: final val bpb $FIRST_BPB, after resuming $RESUMED_BPB (should match closely)," \
        "peak memory $(grep "Peak memory usage:" "$LOG" | tail -1 | awk '{print $NF}')"
done

log "Smoke test passed for: ${ARMS[*]}"
