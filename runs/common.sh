#!/bin/bash

# Shared environment and helpers for the reference ("cheat sheet") experiment. Sourced, not run:
#
#   runs/data.sh     build the training data: packed ClimbMix rows + their retrieved Wikipedia refs
#   runs/smoke.sh    tiny end-to-end check of both arms (minutes), run this before anything long
#   runs/isoflop.sh  IsoFLOP sweep: a few depths per arm at fixed FLOP budgets, to find each arm's best depth
#   runs/train.sh    one headline run of one arm (pretrain + eval)
#
# The two arms train on the exact same packed rows, in the same order:
#   baseline     decoder-only, the row's 8x256 refs concatenated before the doc as a prefix, loss only on the doc
#   cheat_sheet  an encoder (depth // 2 layers) reads the refs, the decoder cross-attends to them in every layer
#
# Everything below can be overridden from the environment, e.g. NPROC=4 bash runs/smoke.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export HF_HUB_OFFLINE=1 # mpnet and Wikipedia are already in the HF cache, never revalidate them over the network
export PYTHONUNBUFFERED=1 # progress reaches tee'd / redirected logs as it happens, not in 8KB blocks
cd "$(dirname "${BASH_SOURCE[0]}")/.." # repo root, so the scripts work from any directory
if [ -d .venv ]; then
    source .venv/bin/activate
elif [ "$(basename "$0")" != data.sh ]; then # (runs/data.sh's setup stage is what creates it)
    echo "No .venv found, run: STAGES=setup bash runs/data.sh"
    exit 1
fi

NPROC="${NPROC:-6}" # GPUs per run. Also the number of packing streams in runs/data.sh, keep the two equal
WANDB_RUN="${WANDB_RUN:-dummy}" # set to enable wandb logging; used as a prefix for each run's name

# One fixed total batch size (in predicted tokens) for every run: then every optimizer step of every run sees
# the same rows regardless of arm, depth or device batch size (nanochat would otherwise size it from the
# param count, which differs between the arms). 491,520 = 40 rows per GPU per step on 6 GPUs.
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-491520}"
# Val tokens for every bpb eval: 1704 rows per GPU (about half the val split) is divisible by device batch
# sizes 1, 2, 4 and 8, so every run evaluates on exactly the same val rows.
EVAL_TOKENS="${EVAL_TOKENS:-$((1704 * 2048 * NPROC))}"

# Flags shared by every pretraining run. Full attention everywhere: the baseline attends over its whole
# 4096-token prefix+doc sequence, cheat_sheet's decoder over its 2048 doc tokens.
COMMON_TRAIN_ARGS=(
    --window-pattern=L
    --max-seq-len=2048
    --total-batch-size="$TOTAL_BATCH_SIZE"
    --eval-tokens="$EVAL_TOKENS"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    log "ERROR: $*"
    exit 1
}

check_arm() {
    case "$1" in
        baseline|cheat_sheet) ;;
        *) die "unknown arm '$1' (expected baseline or cheat_sheet)" ;;
    esac
}

# base_train flags that select an arm's architecture
arm_args() {
    case "$1" in
        baseline) echo "--ref-mode=prefix" ;;
        cheat_sheet) echo "--ref-mode=encoder --n-enc-layer=${N_ENC_LAYER:--1} --cross-every=${CROSS_EVERY:-1}" ;;
    esac
}

# Per-GPU micro-batch (rows) for a 24GB 4090. Gradient accumulation makes up the difference to
# TOTAL_BATCH_SIZE, so this only changes speed, never the result. Override with DEVICE_BATCH_SIZE=N if a
# run OOMs anyway (e.g. another process is sharing the GPU).
#
# Measured with runs/smoke.sh at the real total batch size, --fp8, 30 steps + evals + resume (2026-09-28):
# baseline d24 b1 peaks at 18.8GiB, b2 OOMs; cheat_sheet d20 b1 peaks at 17.6GiB, d22 OOMs even at b1. The
# smaller depths are unmeasured guesses. Note that short probes (1 micro-step, no eval) read about 3GiB low.
# cheat_sheet's encoder + cross-attention make it use about as much memory as a baseline 4 layers deeper
# (which is also about where their scaling params match: cheat_sheet d20 763M vs baseline d24 730M).
device_batch_size() {
    local arm=$1 depth=$2
    [ "$arm" = cheat_sheet ] && depth=$((depth + 4))
    if [ -n "$DEVICE_BATCH_SIZE" ]; then echo "$DEVICE_BATCH_SIZE"
    elif [ "$depth" -le 12 ]; then echo 8
    elif [ "$depth" -le 16 ]; then echo 4
    elif [ "$depth" -le 22 ]; then echo 2
    else echo 1
    fi
}

wandb_name() {
    if [ "$WANDB_RUN" = dummy ]; then echo dummy; else echo "${WANDB_RUN}_$1"; fi
}

# The GPUs are sometimes shared with other jobs: refuse to launch onto ones that already hold a lot of memory
require_free_gpus() {
    [ "${FORCE:-0}" = 1 ] && return 0
    local busy
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', *' -v n="$NPROC" '$1 < n && $2 > 2000 {printf "%s (%s MiB) ", $1, $2}')
    [ -z "$busy" ] || die "GPUs busy: ${busy}-- wait for them, or set FORCE=1 to launch anyway"
}

require_data() {
    local split
    for split in train val; do
        [ -f "$NANOCHAT_BASE_DIR/packed/.${split}_dedup.done" ] || die "the packed $split data isn't finished, run: bash runs/data.sh"
    done
}

launch() {
    torchrun --standalone --nproc_per_node="$NPROC" -m "$@"
}

# run_logged LOGFILE CMD...: run CMD with its output also appended to LOGFILE, and stop everything if it fails
run_logged() {
    local logfile=$1
    shift
    "$@" 2>&1 | tee -a "$logfile"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "failed (log: $logfile): $*"
}

# Logs and results.csv files of all the scripts go here (checkpoints go to base_checkpoints/<model tag> as usual)
EXPERIMENTS_DIR="$NANOCHAT_BASE_DIR/experiments"

# Summary numbers from a base_train log, as CSV fields
SUMMARY_HEADER="model_dim,params_total,params_encoder,flops_per_token,num_iterations,tokens_trained,val_bpb,peak_mem_mib,train_time_min"
summarize_log() {
    local f=$1
    local model_dim params_total params_encoder flops_per_token num_iters tokens val_bpb peak_mem train_time
    model_dim=$(grep '"n_embd":' "$f" | head -1 | grep -oP '\d+')
    params_total=$(grep "^total " "$f" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    params_encoder=$(grep "^encoder " "$f" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    flops_per_token=$(grep "Estimated FLOPs per token:" "$f" | tail -1 | awk '{print $NF}')
    num_iters=$(grep "number of iterations" "$f" | tail -1 | sed 's/.*: //' | tr -d ',')
    tokens=$(grep "Total number of training tokens:" "$f" | tail -1 | sed 's/.*: //' | tr -d ',')
    val_bpb=$(grep "Validation bpb:" "$f" | tail -1 | grep -oP '[\d.]+$')
    peak_mem=$(grep "Peak memory usage:" "$f" | tail -1 | grep -oP '[\d.]+(?=MiB)')
    train_time=$(grep "Total training time:" "$f" | tail -1 | grep -oP '[\d.]+(?=m)')
    echo "$model_dim,$params_total,$params_encoder,$flops_per_token,$num_iters,$tokens,$val_bpb,$peak_mem,$train_time"
}
