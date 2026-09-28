#!/bin/bash

# Build the training data for the reference ("cheat sheet") experiment, in stages:
#
#   setup          venv, plus check/download the raw inputs (ClimbMix shards, tokenizer, cached Wikipedia)
#   wiki_build     tokenize + chunk all of Wikipedia into 256-token chunks                      CPU
#   pack           pack ClimbMix into 2049-token rows exactly as the live loader would          CPU, per split
#   wiki_embed     embed every wiki chunk with mpnet                                            GPU, one process per GPU
#   embed_windows  embed each packed row's 8 windows of 256 tokens with mpnet                   GPU, per split
#   search         top distinct-article wiki candidates for every window                        GPU (index sharded), per split
#   dedup          article-level dedup per row, then fill the rows' 8x256 reference columns     CPU, per split
#   task_refs      8 distinct-article wiki refs per SFT / eval conversation, on its question    GPU (index on 3, mpnet on 1)
#
# Each stage leaves a marker when it completes, so rerunning this script skips finished work and continues where
# it stopped. The two embed stages also resume mid-way after a crash. Re-packing a split deletes that split's
# downstream outputs, since they'd be stale. Output: $NANOCHAT_BASE_DIR/packed/{train,val}.npy (+ wiki/ and
# retrieval side files), read by the packed loader in nanochat/dataloader.py. $NANOCHAT_BASE_DIR/task_refs/ holds
# the SFT / eval conversations' refs, attached with tasks.common.attach_refs.
#
# Usage:
#   bash runs/data.sh                                  # every stage that isn't done yet
#   screen -L -Logfile runs/data.log -S data bash runs/data.sh
#   STAGES="pack" bash runs/data.sh                    # a subset (CPU-only stages don't need free GPUs)
#   SPLITS="val" bash runs/data.sh                     # val only: small, a quick end-to-end check of the pipeline
#
# Afterwards, inspect the retrievals with: python -m scripts.wiki_view --split val [--stats]

source "$(dirname "$0")/common.sh"

STAGES="${STAGES:-setup wiki_build pack wiki_embed embed_windows search dedup task_refs}"
SPLITS="${SPLITS:-val train}"
PACK_DIR="$NANOCHAT_BASE_DIR/packed"
WIKI_DIR="$NANOCHAT_BASE_DIR/wiki"

want() {
    [[ " $STAGES " == *" $1 "* ]]
}

need() {
    [ -f "$1" ] || die "missing $1 -- run the stage that produces it first"
}

# stage MARKER CMD...: run CMD unless MARKER exists, then create MARKER
stage() {
    local marker=$1
    shift
    if [ -f "$marker" ]; then
        log "skip (already done): $*"
        return
    fi
    log "start: $*"
    "$@" || die "failed: $*"
    touch "$marker"
    log "done: $*"
}

# One process per GPU, each handling its shard of the rows (see nanochat.wiki.embed_rows_to_memmap).
# Background jobs ignore Ctrl-C in a script, so take them down with us if we exit early.
trap 'kill $(jobs -p) 2> /dev/null' EXIT
per_gpu() {
    local pids=() r failed=0
    for ((r = 0; r < NPROC; r++)); do
        CUDA_VISIBLE_DEVICES=$r "$@" --rank "$r" --world-size "$NPROC" --gpu 0 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    return $failed
}

pack_split() {
    local split=$1
    rm -f "$PACK_DIR/$split.npy" "$PACK_DIR/${split}"_*.npy "$PACK_DIR/.${split}"_*.done # stale downstream outputs
    python -m nanochat.pack --split "$split" --world-size "$NPROC"
}

# search-shard runs one process per GPU (each resident on its own contiguous slice of the wiki index,
# so the GPU work is fully parallel); search-merge then combines the per-GPU shards into the final
# per-window candidates. Splitting the stage this way is what makes it parallel instead of the previous
# single-process version, which held the whole index but visited its GPU shards one at a time.
search_split() {
    local split=$1
    per_gpu python -m scripts.wiki_retrieve search-shard --split "$split" || return 1
    python -m scripts.wiki_retrieve search-merge --split "$split" --world-size "$NPROC"
}

# -----------------------------------------------------------------------------

if want setup; then
    log "setup: venv + raw inputs"
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync --extra gpu || die "uv sync failed"
    source .venv/bin/activate
    python -m nanochat.dataset -n 170 || die "ClimbMix download failed" # 170 train shards + val, skips ones already present
    [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || python -m scripts.tok_train || die "tokenizer training failed"
    python -c "from nanochat.wiki import list_wiki_source_files; list_wiki_source_files()" || die "Wikipedia not in the HF cache"
fi

if want wiki_build; then
    stage "$WIKI_DIR/config.json" python -m nanochat.wiki build # config.json is written last, so it doubles as the marker
fi

if want pack; then
    for split in $SPLITS; do
        stage "$PACK_DIR/.${split}_pack.done" pack_split "$split"
    done
fi

if want wiki_embed; then
    need "$WIKI_DIR/config.json"
    [ -f "$WIKI_DIR/.embed.done" ] || require_free_gpus
    stage "$WIKI_DIR/.embed.done" per_gpu python -m nanochat.wiki embed
fi

if want embed_windows; then
    for split in $SPLITS; do
        need "$PACK_DIR/.${split}_pack.done"
        [ -f "$PACK_DIR/.${split}_qemb.done" ] || require_free_gpus
        stage "$PACK_DIR/.${split}_qemb.done" per_gpu python -m scripts.wiki_retrieve embed-windows --split "$split"
    done
fi

if want search; then
    for split in $SPLITS; do
        need "$WIKI_DIR/.embed.done"
        need "$PACK_DIR/.${split}_qemb.done"
        [ -f "$PACK_DIR/.${split}_search.done" ] || require_free_gpus
        stage "$PACK_DIR/.${split}_search.done" search_split "$split"
    done
fi

if want dedup; then
    for split in $SPLITS; do
        need "$PACK_DIR/.${split}_search.done"
        stage "$PACK_DIR/.${split}_dedup.done" python -m scripts.wiki_retrieve dedup --split "$split"
    done
fi

if want task_refs; then
    need "$WIKI_DIR/.embed.done"
    [ -f "$NANOCHAT_BASE_DIR/task_refs/.done" ] || require_free_gpus
    stage "$NANOCHAT_BASE_DIR/task_refs/.done" python -m scripts.wiki_retrieve tasks
fi

log "Data stages finished: $STAGES (splits: $SPLITS)"
ls -la "$PACK_DIR"
