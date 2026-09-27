"""
Offline row packing: pre-pack the pretraining corpus into fixed 2049-token rows,
using the exact same best-fit algorithm nanochat's live dataloader uses (see
nanochat.dataloader.BestFitPacker), so the wiki-retrieval pipeline in nanochat/wiki.py
has a fixed target to build references against instead of retrieving on the fly.

Runs `world_size` independent packing streams in parallel (one process per stream,
same rank/shard-of-row-groups convention as the live loader), each writing its own
raw token file for (just under) one epoch, then interleaves them into one npy so that
row i*world_size+r is stream r's i-th row -- i.e. the i-th row a `world_size`-GPU live
loader run would pack on GPU rank r (with device batch size B, step s on rank r is
that rank's rows s*B..s*B+B-1).

    python -m nanochat.pack --split train --world-size 6
    python -m nanochat.pack --split val --world-size 6

Output: <base_dir>/packed/{train,val}.npy, shape (num_rows, ROW_WIDTH) uint16.
Columns [0:DOC_WIDTH) are the packed document tokens (T+1=2049). Columns
[DOC_WIDTH:ROW_WIDTH) are the 8x256 wiki reference tokens, left as zero here --
they're filled in later by scripts/wiki_retrieve.py.
"""
import os
import argparse
from multiprocessing import Pool

import numpy as np

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import _document_batches, BestFitPacker

DOC_WIDTH = 2049       # T+1
NUM_WINDOWS = 8
WINDOW_LEN = 256
ROW_WIDTH = DOC_WIDTH + NUM_WINDOWS * WINDOW_LEN  # 4097

PACK_DIR = os.path.join(get_base_dir(), "packed")

def packed_path(split):
    return os.path.join(PACK_DIR, f"{split}.npy")

def _pack_stream_to_file(split, rank, world_size, out_path, buffer_size, max_rows, flush_rows):
    """ Pack one rank's stream for one epoch, writing raw uint16 rows to out_path. Stops at the
    first second-epoch refill, so the ~buffer_size first-epoch docs still buffered are dropped. """
    tokenizer = get_tokenizer()
    batches = _document_batches(split, resume_state_dict=None, tokenizer_batch_size=128,
                                 rank=rank, world_size=world_size)
    packer = BestFitPacker(batches, tokenizer, buffer_size=buffer_size)

    n_rows = 0
    buf = []
    with open(out_path, "wb") as f:
        while max_rows is None or n_rows < max_rows:
            row = packer.pack_row(DOC_WIDTH)
            if packer.epoch > 1:
                break  # this row would mix in data from a second epoch -- stop before it
            buf.append(np.asarray(row, dtype=np.uint16))
            n_rows += 1
            if len(buf) >= flush_rows:
                np.concatenate(buf).tofile(f)
                buf = []
        if buf:
            np.concatenate(buf).tofile(f)
    return n_rows

def pack(split, world_size=6, buffer_size=1000, max_rows=None, flush_rows=50_000, out_dir=None):
    """ Pack `split` into <out_dir>/{split}.npy using `world_size` parallel streams. """
    out_dir = PACK_DIR if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    tmp_paths = [os.path.join(out_dir, f".{split}_rank{r}.raw") for r in range(world_size)]

    with Pool(world_size) as pool:
        counts = pool.starmap(_pack_stream_to_file, [
            (split, r, world_size, tmp_paths[r], buffer_size, max_rows, flush_rows) for r in range(world_size)])
    n_rows_per_stream = min(counts)
    print(f"Packed {counts} rows per stream, keeping {n_rows_per_stream} rows/stream (trimmed to the shortest)")

    total_rows = n_rows_per_stream * world_size
    out_path = os.path.join(out_dir, f"{split}.npy")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total_rows, ROW_WIDTH))
    for r in range(world_size):
        stream = np.memmap(tmp_paths[r], dtype=np.uint16, mode="r", shape=(counts[r], DOC_WIDTH))
        out[r:total_rows:world_size, :DOC_WIDTH] = stream[:n_rows_per_stream]
    out.flush()

    for p in tmp_paths:
        os.remove(p)
    print(f"Wrote {total_rows} rows to {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline best-fit row packing (matches the live training loader)")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--world-size", type=int, default=6, help="Parallel packing streams (matches GPU count at train time)")
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows per stream, for a quick test run")
    args = parser.parse_args()
    pack(args.split, world_size=args.world_size, buffer_size=args.buffer_size, max_rows=args.max_rows)
