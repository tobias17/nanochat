#!/bin/bash

# Headline run of one arm: pretrain at a fixed FLOP budget (use the same budget for both arms, and each arm's
# best depth from runs/isoflop.sh), evaluate, then SFT and evaluate the chat model.
#
#   PLAN=1 bash runs/train.sh baseline 16                      # CPU only: steps, tokens and epochs of packed data, then exit
#   bash runs/train.sh baseline 16
#   bash runs/train.sh cheat_sheet 18
#   TARGET_FLOPS=3e19 bash runs/train.sh cheat_sheet 18
#   RESUME_STEP=6000 bash runs/train.sh cheat_sheet 18         # continue from a saved checkpoint after a crash
#   STAGES=eval bash runs/train.sh cheat_sheet 18              # only (re)run some stages
#   screen -L -Logfile runs/train_cheat_sheet.log -S cheat_sheet bash runs/train.sh cheat_sheet 18
#
# Stages (STAGES="pretrain eval sft" by default):
#   pretrain  base_train, checkpointing every SAVE_EVERY steps to base_checkpoints/<arm>_d<depth>
#   eval      base_eval: train/val bpb on the packed rows. CORE and sampling are skipped for now: for the
#             reference arms they need refs retrieved for their prompts at eval time, which isn't wired up yet
#   sft       chat_sft (one epoch of SmolTalk + MMLU + GSM8K, one conversation per row, each with its 8 wiki refs
#             precomputed by data.sh task_refs) to chatsft_checkpoints/<tag>, then chat_eval: ChatCORE on
#             ARC-E/C, MMLU, GSM8K, HumanEval, also with precomputed refs
#
# Dry run (the whole pipeline at full size, just short): a small TARGET_FLOPS gives few steps, e.g.
#   TARGET_FLOPS=3e17 EVAL_EVERY_FLOPS=1e17 SAVE_EVERY=20 SFT_ITERATIONS=20 CHAT_EVAL_MAX_PROBLEMS=48 TAG=dry_baseline_d24 \
#       bash runs/train.sh baseline 24
#
# Logs and results.csv: $NANOCHAT_BASE_DIR/experiments/train/
# Per-step metrics: base_checkpoints/<tag>/metrics.jsonl. Compare the arms on the FLOPs axis with e.g.
#   python -m scripts.compare_runs baseline_d16 cheat_sheet_d18 [--plot compare.png]

source "$(dirname "$0")/common.sh"

ARM="$1"
DEPTH="$2"
[ -n "$ARM" ] && [ -n "$DEPTH" ] || die "usage: bash runs/train.sh <baseline|cheat_sheet> <depth>"
check_arm "$ARM"

TARGET_FLOPS="${TARGET_FLOPS:-2e19}" # the old d24 decoder-only run was about 3e19 (13h on 6 4090s)
EVAL_EVERY_FLOPS="${EVAL_EVERY_FLOPS:-5e17}" # val bpb at the same compute points in every run (40 evals at 2e19), see scripts/compare_runs.py
STAGES="${STAGES:-pretrain eval sft}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
TAG="${TAG:-${ARM}_d${DEPTH}}"
DBS=$(device_batch_size "$ARM" "$DEPTH")
FP8_ARG=$([ "${FP8:-1}" = 1 ] && echo --fp8) # the 4090 supports FP8 and the old d24 run used it
RESUME_ARG=$([ -n "$RESUME_STEP" ] && echo "--resume-from-step=$RESUME_STEP")
SFT_ITERS_ARG=$([ -n "$SFT_ITERATIONS" ] && echo "--num-iterations=$SFT_ITERATIONS") # default: one epoch
CHAT_EVAL_MAX_ARG=$([ -n "$CHAT_EVAL_MAX_PROBLEMS" ] && echo "--max-problems=$CHAT_EVAL_MAX_PROBLEMS") # default: all

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
    --eval-every-flops="$EVAL_EVERY_FLOPS"
    --model-tag="$TAG"
)
if [ "${PLAN:-0}" = 1 ]; then
    CUDA_VISIBLE_DEVICES= python -m scripts.base_train "${TRAIN_ARGS[@]}" --device-type=cpu --plan
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
    [ -f "$NANOCHAT_BASE_DIR/task_refs/.done" ] || die "the SFT / eval refs aren't finished, run: bash runs/data.sh"
    run_logged "$OUT_DIR/${TAG}_sft.log" launch scripts.chat_sft -- \
        --model-tag="$TAG" \
        --device-batch-size="$DBS" \
        --run="$(wandb_name "${TAG}_sft")" $SFT_ITERS_ARG
    CHAT_EVAL_LOG="$OUT_DIR/${TAG}_chat_eval.log"
    run_logged "$CHAT_EVAL_LOG" launch scripts.chat_eval -- --source=sft --model-tag="$TAG" $CHAT_EVAL_MAX_ARG
    log "Finished SFT of $TAG: $(grep -E "accuracy:|ChatCORE metric:" "$CHAT_EVAL_LOG" | tail -6 | tr '\n' ' ')"
fi

log "Done: $TAG ($STAGES)"
