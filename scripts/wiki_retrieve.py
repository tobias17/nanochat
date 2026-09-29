"""
Join packed pretraining rows (see nanochat.pack) to Wikipedia, per 256-token window,
never repeating a chunk's text: no two of a row's 8 windows may end up referencing
the same chunk (or byte-identical copies of it, see nanochat.wiki.chunk_content_ids); different
chunks of one article are fine. Three offline steps, run after nanochat.wiki
build + embed:

    python -m scripts.wiki_retrieve embed-windows --split train
    python -m scripts.wiki_retrieve search-shard --split train   # one process per GPU (see runs/data.sh's per_gpu)
    python -m scripts.wiki_retrieve search-merge --split train
    python -m scripts.wiki_retrieve dedup --split train

and one more for SFT / eval conversations, which get their 8 nearest distinct-text chunks each, retrieved on the
question / user prompt (see Task.retrieval_query):

    python -m scripts.wiki_retrieve tasks   # writes <base_dir>/task_refs/{ref_key}_{ids,scores}.npy + {ref_key}.json

Writes, under <base_dir>/packed/:
  {split}_qemb.npy         (R*8, dim) fp16    -- window embeddings
  {split}_cand_ids.npy      (R, 8, 32) int32   -- top distinct-text wiki chunk ids per window, score-sorted
  {split}_cand_scores.npy   (R, 8, 32) float32
  {split}_ref_ids.npy       (R, 8) int32       -- final deduped chunk id per window
  {split}_ref_scores.npy    (R, 8) float32
and fills the wiki-reference columns of {split}.npy in place.
"""
import os
import json
import time
import argparse
import itertools

import numpy as np

from nanochat.tokenizer import get_tokenizer
from nanochat.pack import PACK_DIR, DOC_WIDTH, NUM_WINDOWS, WINDOW_LEN, packed_path
from nanochat.wiki import WIKI_DIR, EMBED_MODEL_NAME, embed_rows_to_memmap, top_distinct, chunk_content_ids

CAND_K = 128     # raw nearest chunks fetched per window before collapsing duplicate texts
DISTINCT_K = 32  # distinct-text candidates kept per window (8 windows can never need more than 8)

def _window_ids(rows, row_idx, w, bos_id):
    start = w * WINDOW_LEN
    ids = rows[row_idx, start:start + WINDOW_LEN].tolist()
    return [t for t in ids if t != bos_id]  # BOS marks document boundaries within the row; not real content

# -----------------------------------------------------------------------------

def embed_windows(split, batch_size=256, rank=0, world_size=6, gpu=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is already cached; don't hit the network to revalidate it
    from sentence_transformers import SentenceTransformer
    import torch

    gpu = rank if gpu is None else gpu
    tokenizer = get_tokenizer()
    bos_id = tokenizer.get_bos_token_id()
    rows = np.load(packed_path(split), mmap_mode="r")
    n = rows.shape[0] * NUM_WINDOWS

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    def row_ids_fn(flat_i):
        row_idx, w = divmod(flat_i, NUM_WINDOWS)
        return _window_ids(rows, row_idx, w, bos_id)

    out_path = os.path.join(PACK_DIR, f"{split}_qemb.npy")
    start, end = embed_rows_to_memmap(
        row_ids_fn, n, out_path, model, tokenizer,
        batch_size=batch_size, dtype="fp16", rank=rank, world_size=world_size, log_prefix=f"windows cuda:{gpu} ",
    )
    print(f"rank {rank} (cuda:{gpu}): done, wrote rows [{start}:{end}) of {out_path}")

# -----------------------------------------------------------------------------

def _shard_paths(split, rank):
    return (os.path.join(PACK_DIR, f".{split}_shard{rank}_idx.npy"),
            os.path.join(PACK_DIR, f".{split}_shard{rank}_scores.npy"))

def search_shard(split, rank, world_size=6, wiki_dtype="fp16", gpu=None, block_size=200_000, batch_size=4096):
    """
    One rank's slice of the search: hold only this rank's contiguous slice of the wiki index resident
    on its own GPU, and find its local top-CAND_K over every window query. world_size ranks run as
    separate processes (see runs/data.sh's per_gpu), one per GPU, so the GPU work is fully parallel --
    the true global top-CAND_K is always found within the union of shards' local top-CAND_K, since a
    globally-top-CAND_K candidate is necessarily also top-CAND_K within its own shard. search_merge
    combines the world_size shards afterward.
    """
    import torch
    from nanochat.wiki import _shard_bounds

    device = f"cuda:{gpu if gpu is not None else rank}"
    rows = np.load(packed_path(split), mmap_mode="r")
    n = rows.shape[0] * NUM_WINDOWS
    qemb = np.load(os.path.join(PACK_DIR, f"{split}_qemb.npy"), mmap_mode="r")
    assert qemb.shape[0] == n, "run embed-windows for every rank first"

    emb_path = os.path.join(WIKI_DIR, "emb_int8.npy" if wiki_dtype == "int8" else "emb.npy")
    wiki_emb = np.load(emb_path, mmap_mode="r")
    row_start, row_end = _shard_bounds(wiki_emb.shape[0], rank, world_size)
    np_dtype = torch.int8 if wiki_dtype == "int8" else torch.float16
    resident = torch.from_numpy(np.array(wiki_emb[row_start:row_end])).to(device, dtype=np_dtype)

    idx_path, scores_path = _shard_paths(split, rank)
    if not os.path.exists(idx_path):
        idx_mm = np.lib.format.open_memmap(idx_path, mode="w+", dtype=np.int32, shape=(n, CAND_K))
        idx_mm[:] = -1  # sentinel: -1 marks a not-yet-computed row (0 is a valid chunk id)
        idx_mm.flush()
    idx_mm = np.lib.format.open_memmap(idx_path, mode="r+")
    scores_mm = np.lib.format.open_memmap(
        scores_path, mode="r+" if os.path.exists(scores_path) else "w+", dtype=np.float32, shape=(n, CAND_K))

    t0 = time.time()
    for bstart in range(0, n, batch_size):
        bend = min(bstart + batch_size, n)
        if idx_mm[bend - 1, 0] != -1:
            continue  # resume: this batch was already computed by an earlier (interrupted) run
        q = torch.from_numpy(np.asarray(qemb[bstart:bend], dtype=np.float32)).to(device, dtype=torch.float16)
        best_scores = torch.full((bend - bstart, CAND_K), -1e9, device=device, dtype=torch.float32)
        best_idx = torch.zeros((bend - bstart, CAND_K), dtype=torch.long, device=device)
        n_rows = resident.shape[0]
        for cstart in range(0, n_rows, block_size):
            cend = min(cstart + block_size, n_rows)
            block = resident[cstart:cend]
            block = (block.to(torch.float16) / 127.0) if wiki_dtype == "int8" else block
            scores = (q @ block.T).float()
            kk = min(CAND_K, scores.shape[1])
            vals, idxs = torch.topk(scores, k=kk, dim=1)
            idxs = idxs + row_start + cstart
            cat_scores = torch.cat([best_scores, vals], dim=1)
            cat_idx = torch.cat([best_idx, idxs], dim=1)
            top_vals, top_pos = torch.topk(cat_scores, k=CAND_K, dim=1)
            best_scores = top_vals
            best_idx = torch.gather(cat_idx, 1, top_pos)
        idx_mm[bstart:bend] = best_idx.cpu().numpy()
        scores_mm[bstart:bend] = best_scores.cpu().numpy()
        if (bstart // batch_size) % 20 == 0 or bend == n:
            rate = bend / max(time.time() - t0, 1e-6)
            print(f"search-shard {device} rank {rank}: {bend}/{n} windows ({rate:.0f}/s)")
    idx_mm.flush()
    scores_mm.flush()
    print(f"rank {rank} ({device}): done, shard [{row_start}:{row_end}) top-{CAND_K}")

def search_merge(split, world_size=6, merge_batch=200_000):
    """ Combine world_size shards' local top-CAND_K into each window's top-DISTINCT_K distinct-text
    candidates (score-sorted), then delete the per-shard temp files. """
    rows = np.load(packed_path(split), mmap_mode="r")
    R = rows.shape[0]
    n = R * NUM_WINDOWS
    content_ids = chunk_content_ids()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"  # full sort of world_size*CAND_K scores is the CPU bottleneck

    shard_idx = [np.load(_shard_paths(split, r)[0], mmap_mode="r") for r in range(world_size)]
    shard_scores = [np.load(_shard_paths(split, r)[1], mmap_mode="r") for r in range(world_size)]

    cand_ids = np.lib.format.open_memmap(
        os.path.join(PACK_DIR, f"{split}_cand_ids.npy"), mode="w+", dtype=np.int32, shape=(R, NUM_WINDOWS, DISTINCT_K))
    cand_scores = np.lib.format.open_memmap(
        os.path.join(PACK_DIR, f"{split}_cand_scores.npy"), mode="w+", dtype=np.float32, shape=(R, NUM_WINDOWS, DISTINCT_K))
    cand_ids_flat, cand_scores_flat = cand_ids.reshape(n, DISTINCT_K), cand_scores.reshape(n, DISTINCT_K)  # one row per window

    t0 = time.time()
    for bstart in range(0, n, merge_batch):
        bend = min(bstart + merge_batch, n)
        idx_cat = np.concatenate([s[bstart:bend] for s in shard_idx], axis=1)      # (b, world_size*CAND_K)
        score_cat = np.concatenate([s[bstart:bend] for s in shard_scores], axis=1)
        top_scores, pos = torch.topk(torch.from_numpy(score_cat).to(device), CAND_K, dim=1)  # sorted descending
        raw_idx = torch.gather(torch.from_numpy(idx_cat).to(device), 1, pos).cpu().numpy()
        raw_scores = top_scores.cpu().numpy()
        batch_ids, batch_scores = top_distinct(raw_idx, raw_scores, content_ids, DISTINCT_K)
        cand_ids_flat[bstart:bend] = batch_ids
        cand_scores_flat[bstart:bend] = batch_scores
        if (bstart // merge_batch) % 5 == 0 or bend == n:
            rate = bend / max(time.time() - t0, 1e-6)
            print(f"search-merge: {bend}/{n} windows ({rate:.0f}/s)")
    cand_ids.flush()
    cand_scores.flush()
    print(f"Wrote per-window distinct-text candidates for {R} rows to {PACK_DIR}")

    for r in range(world_size):
        for p in _shard_paths(split, r):
            os.remove(p)

# -----------------------------------------------------------------------------

def dedup_row(cand_ids_row, cand_scores_row, content_ids):
    """
    Distinct-text dedup for one row's candidate lists.

    cand_ids_row/cand_scores_row: (num_windows, K), score-sorted descending per window,
    -1 id = no more candidates. content_ids: chunk_id -> id of its text (identical texts share one).

    Each window starts at its best candidate; whenever two windows share a text, the
    lower-scoring one advances to its next candidate. Repeats until no two windows share
    a text. Terminates because a window's candidate cursor only moves forward.
    Returns (chunk_ids, scores), each length num_windows (-1/-1e9 if a window ran out of
    candidates, which shouldn't happen once K is comfortably larger than num_windows).
    """
    num_windows, k = cand_ids_row.shape
    cursor = [0] * num_windows
    chunk = [int(cand_ids_row[w, 0]) for w in range(num_windows)]
    score = [float(cand_scores_row[w, 0]) for w in range(num_windows)]
    content = [int(content_ids[c]) if c >= 0 else -1 for c in chunk]
    pairs = list(itertools.combinations(range(num_windows), 2))

    while True:
        conflict = next((p for p in pairs if content[p[0]] == content[p[1]] and content[p[0]] != -1), None)
        if conflict is None:
            break
        a, b = conflict
        loser = a if score[a] < score[b] else b
        cursor[loser] += 1
        j = cursor[loser]
        if j < k and cand_ids_row[loser, j] >= 0:
            chunk[loser] = int(cand_ids_row[loser, j])
            score[loser] = float(cand_scores_row[loser, j])
            content[loser] = int(content_ids[chunk[loser]])
        else:
            chunk[loser], score[loser], content[loser] = -1, -1e9, -1  # ran out of candidates
    return chunk, score

def dedup_and_fill(split):
    rows = np.lib.format.open_memmap(packed_path(split), mode="r+")
    R = rows.shape[0]
    cand_ids = np.load(os.path.join(PACK_DIR, f"{split}_cand_ids.npy"), mmap_mode="r")
    cand_scores = np.load(os.path.join(PACK_DIR, f"{split}_cand_scores.npy"), mmap_mode="r")
    content_ids = chunk_content_ids()
    wiki_tokens = np.load(os.path.join(WIKI_DIR, "tokens.npy"), mmap_mode="r")

    ref_ids = np.full((R, NUM_WINDOWS), -1, dtype=np.int32)
    ref_scores = np.full((R, NUM_WINDOWS), -1e9, dtype=np.float32)

    t0 = time.time()
    n_displaced = 0
    for row_idx in range(R):
        chunk, score = dedup_row(cand_ids[row_idx], cand_scores[row_idx], content_ids)
        for w in range(NUM_WINDOWS):
            ref_ids[row_idx, w] = chunk[w]
            ref_scores[row_idx, w] = score[w]
            if chunk[w] != int(cand_ids[row_idx, w, 0]):
                n_displaced += 1
            col = DOC_WIDTH + w * WINDOW_LEN
            if chunk[w] >= 0:
                rows[row_idx, col:col + WINDOW_LEN] = wiki_tokens[chunk[w]]
        if row_idx % 100_000 == 0 or row_idx == R - 1:
            rate = (row_idx + 1) / max(time.time() - t0, 1e-6)
            print(f"dedup: {row_idx + 1}/{R} rows ({rate:.0f}/s, {n_displaced} displacements so far)")

    rows.flush()
    np.save(os.path.join(PACK_DIR, f"{split}_ref_ids.npy"), ref_ids)
    np.save(os.path.join(PACK_DIR, f"{split}_ref_scores.npy"), ref_scores)
    print(f"Deduped {R} rows ({n_displaced} total displacements), filled wiki columns in {packed_path(split)}")

# -----------------------------------------------------------------------------
# SFT / eval conversations

def _ref_tasks():
    from tasks.smoltalk import SmolTalk
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K
    from tasks.arc import ARC
    from tasks.humaneval import HumanEval
    return {
        "humaneval_test": lambda: HumanEval(),
        "gsm8k_main_test": lambda: GSM8K("main", "test"),
        "arc_ARC-Easy_test": lambda: ARC("ARC-Easy", "test"),
        "arc_ARC-Challenge_test": lambda: ARC("ARC-Challenge", "test"),
        "mmlu_all_test": lambda: MMLU("all", "test"),
        "smoltalk_test": lambda: SmolTalk("test"),
        "gsm8k_main_train": lambda: GSM8K("main", "train"),
        "mmlu_all_auxiliary_train": lambda: MMLU("all", "auxiliary_train"),
        "smoltalk_train": lambda: SmolTalk("train"),
    }  # smallest first, so problems show up early

def retrieve_tasks(keys=None, index_gpus=(0, 1, 2), embed_gpu=3, n_ref=8, batch_size=256, force=False):
    """ Retrieve the n_ref nearest distinct-text wiki chunks for every example of each task, in one process.
    Keys whose {key}.json exists are skipped unless force, so a crashed run resumes. """
    from nanochat.wiki import Retriever
    from tasks.common import REFS_DIR, queries_hash

    ref_tasks = _ref_tasks()
    keys = list(ref_tasks) if not keys else keys
    for key in keys:
        assert key in ref_tasks, f"unknown key {key}, choose from {list(ref_tasks)}"
    os.makedirs(REFS_DIR, exist_ok=True)
    meta_path = lambda key: os.path.join(REFS_DIR, f"{key}.json")
    if not force:
        for key in [k for k in keys if os.path.exists(meta_path(k))]:
            print(f"{key}: already done, skipping (--force to redo)", flush=True)
        keys = [k for k in keys if not os.path.exists(meta_path(k))]
    if not keys:
        return
    retriever = Retriever(list(index_gpus), f"cuda:{embed_gpu}", n_ref=n_ref, cand_k=CAND_K)

    for key in keys:
        if os.path.exists(meta_path(key)):
            os.remove(meta_path(key))  # a crash mid-rewrite must not leave the old marker next to new arrays
        task = ref_tasks[key]()
        assert task.ref_key == key, f"{key} vs task.ref_key {task.ref_key}"
        n = task.num_examples()
        t0 = time.time()
        queries = [task.retrieval_query(i) for i in range(n)]
        ids, scores = retriever.retrieve(queries, batch_size=batch_size)
        n_short = int((ids < 0).any(axis=1).sum())
        np.save(os.path.join(REFS_DIR, f"{key}_ids.npy"), ids)
        np.save(os.path.join(REFS_DIR, f"{key}_scores.npy"), scores)
        meta = {"n": n, "n_ref": n_ref, "embed_model": EMBED_MODEL_NAME, "n_chunks": int(retriever.index.n),
                "queries_sha256": queries_hash(task)}
        with open(meta_path(key), "w") as f:  # written last: doubles as the completion marker
            json.dump(meta, f)
        valid = scores[scores > -1e8]
        print(f"{key}: {n} queries in {time.time() - t0:.0f}s ({n / max(time.time() - t0, 1e-6):.0f}/s), "
              f"{n_short} with <{n_ref} refs, score p5/p50/p95 = {np.percentile(valid, [5, 50, 95]).round(3).tolist()}", flush=True)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join packed rows' windows to Wikipedia, never repeating a chunk's text")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ew = sub.add_parser("embed-windows", help="Embed every window of every packed row")
    p_ew.add_argument("--split", choices=["train", "val"], required=True)
    p_ew.add_argument("--batch-size", type=int, default=256)
    p_ew.add_argument("--rank", type=int, default=0)
    p_ew.add_argument("--world-size", type=int, default=6)
    p_ew.add_argument("--gpu", type=int, default=None)

    p_ss = sub.add_parser("search-shard", help="One rank's shard of the search: local top-K candidates on its own GPU")
    p_ss.add_argument("--split", choices=["train", "val"], required=True)
    p_ss.add_argument("--dtype", choices=["fp16", "int8"], default="fp16")
    p_ss.add_argument("--block-size", type=int, default=200_000)
    p_ss.add_argument("--batch-size", type=int, default=4096)
    p_ss.add_argument("--rank", type=int, default=0)
    p_ss.add_argument("--world-size", type=int, default=6)
    p_ss.add_argument("--gpu", type=int, default=None)

    p_sm = sub.add_parser("search-merge", help="Merge shards' local top-K into each window's top distinct-text candidates")
    p_sm.add_argument("--split", choices=["train", "val"], required=True)
    p_sm.add_argument("--world-size", type=int, default=6)

    p_dedup = sub.add_parser("dedup", help="Resolve repeated-chunk conflicts and fill the packed rows' wiki columns")
    p_dedup.add_argument("--split", choices=["train", "val"], required=True)

    p_t = sub.add_parser("tasks", help="Retrieve wiki refs for SFT / eval task conversations")
    p_t.add_argument("--keys", nargs="*", default=None, help="Task ref keys to do (default: all)")
    p_t.add_argument("--force", action="store_true", help="Redo keys that are already done")
    p_t.add_argument("--index-gpus", type=int, nargs="+", default=[0, 1, 2], help="GPUs holding the wiki index (fp16 needs about 3 x 24GB)")
    p_t.add_argument("--embed-gpu", type=int, default=3, help="GPU for the query embedding model")
    p_t.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()
    if args.command == "embed-windows":
        embed_windows(args.split, batch_size=args.batch_size, rank=args.rank, world_size=args.world_size, gpu=args.gpu)
    elif args.command == "search-shard":
        search_shard(args.split, rank=args.rank, world_size=args.world_size, wiki_dtype=args.dtype,
                     gpu=args.gpu, block_size=args.block_size, batch_size=args.batch_size)
    elif args.command == "search-merge":
        search_merge(args.split, world_size=args.world_size)
    elif args.command == "dedup":
        dedup_and_fill(args.split)
    elif args.command == "tasks":
        retrieve_tasks(args.keys, args.index_gpus, args.embed_gpu, batch_size=args.batch_size, force=args.force)
