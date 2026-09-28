#!/bin/bash

# IsoFLOP sweep for the reference experiment: at each FLOP budget, train each arm at a few depths with
# --target-flops, then compare the two arms at their own best depth. The FLOPs count everything an arm spends
# per predicted token (the baseline's ref prefix, cheat_sheet's encoder and cross-attention; see
# GPT.estimate_flops), and every run uses the same batch size on the same rows.
#
#   DRY_RUN=1 bash runs/isoflop.sh     # CPU only, no data needed: params, FLOPs/token, steps, tokens and epochs
#                                      # of every run in the sweep, to choose depths/budgets before spending GPU time
#   bash runs/isoflop.sh               # the sweep. Finished runs are skipped, so it can simply be restarted
#   LABEL=v2 FLOPS_BUDGETS="1e18" DEPTHS_cheat_sheet="10 12" bash runs/isoflop.sh
#   screen -L -Logfile runs/isoflop.log -S isoflop bash runs/isoflop.sh
#
# Results: $NANOCHAT_BASE_DIR/experiments/isoflop_<label>/results.csv, plus one log per run.
# Checkpoints (final step only): base_checkpoints/isoflop_<label>_<flops>_<arm>_d<depth>

source "$(dirname "$0")/common.sh"

LABEL="${LABEL:-sweep1}"
FLOPS_BUDGETS=(${FLOPS_BUDGETS:-1e18 3e18})
ARMS=(${ARMS:-baseline cheat_sheet})
DEPTHS="${DEPTHS:-8 10 12 14}" # per arm overrides: DEPTHS_baseline="...", DEPTHS_cheat_sheet="..."
FP8_ARG=$([ "${FP8:-0}" = 1 ] && echo --fp8) # off by default: at these small widths most matmuls are too small to gain much

OUT_DIR="$EXPERIMENTS_DIR/isoflop_${LABEL}"
RESULTS_FILE="$OUT_DIR/results.csv"
if [ "${DRY_RUN:-0}" != 1 ]; then
    require_data
    require_free_gpus
    mkdir -p "$OUT_DIR"
    [ -f "$RESULTS_FILE" ] || echo "flops_budget,arm,depth,device_batch_size,$SUMMARY_HEADER" > "$RESULTS_FILE"
else
    printf "%-8s %-12s %5s %14s %14s %12s %8s %16s %s\n" flops arm depth params encoder flops/tok steps tokens epochs
fi

for flops in "${FLOPS_BUDGETS[@]}"; do
    for arm in "${ARMS[@]}"; do
        check_arm "$arm"
        depths_var="DEPTHS_${arm}"
        for d in ${!depths_var:-$DEPTHS}; do
            TAG="isoflop_${LABEL}_${flops}_${arm}_d${d}"
            TRAIN_ARGS=(
                "${COMMON_TRAIN_ARGS[@]}" $(arm_args "$arm")
                --depth="$d"
                --target-flops="$flops"
                --target-param-data-ratio=-1
                --model-tag="$TAG"
            )

            if [ "${DRY_RUN:-0}" = 1 ]; then
                OUT=$(CUDA_VISIBLE_DEVICES= python -m scripts.base_train "${TRAIN_ARGS[@]}" --device-type=cpu --dry-run 2>&1) \
                    || { echo "$OUT"; die "dry run failed: $TAG"; }
                printf "%-8s %-12s %5s %14s %14s %12s %8s %16s %s\n" "$flops" "$arm" "$d" \
                    "$(grep "^total " <<< "$OUT" | awk '{print $NF}')" \
                    "$(grep "^encoder " <<< "$OUT" | awk '{print $NF}')" \
                    "$(grep "Estimated FLOPs per token:" <<< "$OUT" | awk '{print $NF}')" \
                    "$(grep "number of iterations" <<< "$OUT" | sed 's/.*: //')" \
                    "$(grep "Total number of training tokens:" <<< "$OUT" | sed 's/.*: //')" \
                    "$(grep -oP '[\d.]+(?= epochs)' <<< "$OUT" || echo "?")"
                continue
            fi

            if grep -q "^${flops},${arm},${d}," "$RESULTS_FILE"; then
                log "Skipping $TAG (already in $RESULTS_FILE)"
                continue
            fi
            DBS=$(device_batch_size "$arm" "$d")
            LOG="$OUT_DIR/${TAG}.log"
            : > "$LOG"
            log "Training $TAG (device batch size $DBS)"
            run_logged "$LOG" launch scripts.base_train -- "${TRAIN_ARGS[@]}" $FP8_ARG \
                --device-batch-size="$DBS" \
                --core-metric-every=-1 \
                --sample-every=-1 \
                --save-every=-1 \
                --run="$(wandb_name "$TAG")"
            SUMMARY=$(summarize_log "$LOG")
            echo "$flops,$arm,$d,$DBS,$SUMMARY" >> "$RESULTS_FILE"
            log "Finished $TAG: $SUMMARY_HEADER = $SUMMARY"
        done
    done
done

if [ "${DRY_RUN:-0}" != 1 ]; then
    log "IsoFLOP sweep complete: $RESULTS_FILE"
    column -t -s',' "$RESULTS_FILE"
fi
