#!/bin/bash

# Headline run of one arm: pretrain at a fixed FLOP budget (use the same budget for both arms, and each arm's
# best depth from runs/isoflop.sh), then evaluate.
#
#   DRY_RUN=1 bash runs/train.sh baseline 16                   # CPU only: steps, tokens and epochs of packed data, then exit
#   bash runs/train.sh baseline 16
#   bash runs/train.sh cheat_sheet 18
#   TARGET_FLOPS=3e19 bash runs/train.sh cheat_sheet 18
#   RESUME_STEP=6000 bash runs/train.sh cheat_sheet 18         # continue from a saved checkpoint after a crash
#   STAGES=eval bash runs/train.sh cheat_sheet 18              # only (re)run some stages
#   screen -L -Logfile runs/train_cheat_sheet.log -S cheat_sheet bash runs/train.sh cheat_sheet 18
#
# Stages (STAGES="pretrain eval" by default):
#   pretrain  base_train, checkpointing every SAVE_EVERY steps to base_checkpoints/<arm>_d<depth>
#   eval      base_eval: train/val bpb on the packed rows. CORE and sampling are skipped for now: for the
#             reference arms they need refs retrieved for their prompts at eval time, which isn't wired up yet
#   sft       placeholder: SFT conversations have precomputed refs (data.sh task_refs), but chat_sft.py doesn't use them yet
#
# Logs and results.csv: $NANOCHAT_BASE_DIR/experiments/train/

source "$(dirname "$0")/common.sh"

ARM="$1"
DEPTH="$2"
[ -n "$ARM" ] && [ -n "$DEPTH" ] || die "usage: bash runs/train.sh <baseline|cheat_sheet> <depth>"
check_arm "$ARM"

TARGET_FLOPS="${TARGET_FLOPS:-2e19}" # the old d24 decoder-only run was about 3e19 (13h on 6 4090s)
STAGES="${STAGES:-pretrain eval}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
TAG="${TAG:-${ARM}_d${DEPTH}}"
DBS=$(device_batch_size "$ARM" "$DEPTH")
FP8_ARG=$([ "${FP8:-1}" = 1 ] && echo --fp8) # the 4090 supports FP8 and the old d24 run used it
RESUME_ARG=$([ -n "$RESUME_STEP" ] && echo "--resume-from-step=$RESUME_STEP")

OUT_DIR="$EXPERIMENTS_DIR/train"
RESULTS_FILE="$OUT_DIR/results.csv"
mkdir -p "$OUT_DIR"
[ -f "$RESULTS_FILE" ] || echo "tag,arm,depth,target_flops,device_batch_size,$SUMMARY_HEADER" > "$RESULTS_FILE"

want() {
    [[ " $STAGES " == *" $1 "* ]]
}

TRAIN_ARGS=(
    "${COMMON_TRAIN_ARGS[@]}" $(arm_args "$ARM")
    --depth="$DEPTH"
    --target-flops="$TARGET_FLOPS"
    --model-tag="$TAG"
)
if [ "${DRY_RUN:-0}" = 1 ]; then
    CUDA_VISIBLE_DEVICES= python -m scripts.base_train "${TRAIN_ARGS[@]}" --device-type=cpu --dry-run
    exit
fi

require_data
require_free_gpus

if want pretrain; then
    LOG="$OUT_DIR/${TAG}_train.log"
    log "Pretraining $TAG to $TARGET_FLOPS FLOPs (device batch size $DBS) ${RESUME_ARG}"
    run_logged "$LOG" launch scripts.base_train -- "${TRAIN_ARGS[@]}" $FP8_ARG $RESUME_ARG \
        --device-batch-size="$DBS" \
        --save-every="$SAVE_EVERY" \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --run="$(wandb_name "$TAG")"
    SUMMARY=$(summarize_log "$LOG")
    echo "$TAG,$ARM,$DEPTH,$TARGET_FLOPS,$DBS,$SUMMARY" >> "$RESULTS_FILE"
    log "Finished pretraining $TAG: $SUMMARY_HEADER = $SUMMARY"
fi

if want eval; then
    run_logged "$OUT_DIR/${TAG}_eval.log" launch scripts.base_eval -- \
        --model-tag="$TAG" \
        --eval=bpb \
        --device-batch-size="$DBS" \
        --split-tokens="$EVAL_TOKENS"
fi

if want sft; then
    die "SFT for the reference arms isn't implemented yet: the refs are precomputed (data.sh task_refs), but chat_sft.py doesn't load them yet"
fi

log "Done: $TAG ($STAGES)"
