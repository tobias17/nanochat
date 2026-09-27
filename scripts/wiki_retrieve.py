"""
Join packed pretraining rows (see nanochat.pack) to Wikipedia, per 256-token window,
with article-level deduplication: no two of a row's 8 windows may end up referencing
chunks from the same Wikipedia article. Three offline steps, run after nanochat.wiki
build + embed:

    python -m scripts.wiki_retrieve embed-windows --split train
    python -m scripts.wiki_retrieve search --split train
    python -m scripts.wiki_retrieve dedup --split train

Writes, under <base_dir>/packed/:
  {split}_qemb.npy         (R*8, dim) fp16    -- window embeddings
  {split}_cand_ids.npy      (R, 8, 32) int32   -- top distinct-article wiki chunk ids per window, score-sorted
  {split}_cand_scores.npy   (R, 8, 32) float32
  {split}_ref_ids.npy       (R, 8) int32       -- final deduped chunk id per window
  {split}_ref_scores.npy    (R, 8) float32
and fills the wiki-reference columns of {split}.npy in place.
"""
import os
import time
import argparse
import itertools

import numpy as np

from nanochat.tokenizer import get_tokenizer
from nanochat.pack import PACK_DIR, DOC_WIDTH, NUM_WINDOWS, WINDOW_LEN, packed_path
from nanochat.wiki import WIKI_DIR, EMBED_MODEL_NAME, WikiIndex, embed_rows_to_memmap

CAND_K = 128    # raw nearest chunks fetched per window before collapsing to distinct articles
ARTICLE_K = 32  # distinct-article candidates kept per window after collapsing (8 windows can never need more than 8)

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

def search(split, wiki_dtype="fp16", gpus=None, block_size=1_000_000, batch_size=4096):
    rows = np.load(packed_path(split), mmap_mode="r")
    R = rows.shape[0]
    n = R * NUM_WINDOWS
    qemb_path = os.path.join(PACK_DIR, f"{split}_qemb.npy")
    qemb = np.load(qemb_path, mmap_mode="r")
    assert qemb.shape[0] == n, (
        f"{qemb_path} has {qemb.shape[0]} rows, expected {n} -- run embed-windows for every rank first"
    )
    chunk_article = np.load(os.path.join(WIKI_DIR, "meta.npy"))[:, 0]  # meta is (n_chunks, 2): [article, chunk_idx]

    index = WikiIndex(dtype=wiki_dtype, gpus=gpus, block_size=block_size)
    print(f"Loaded wiki index: {index.n} chunks across {len(index.shards)} GPU shard(s)")

    cand_ids = np.lib.format.open_memmap(
        os.path.join(PACK_DIR, f"{split}_cand_ids.npy"), mode="w+", dtype=np.int32, shape=(R, NUM_WINDOWS, ARTICLE_K))
    cand_scores = np.lib.format.open_memmap(
        os.path.join(PACK_DIR, f"{split}_cand_scores.npy"), mode="w+", dtype=np.float32, shape=(R, NUM_WINDOWS, ARTICLE_K))
    cand_ids_flat, cand_scores_flat = cand_ids.reshape(n, ARTICLE_K), cand_scores.reshape(n, ARTICLE_K)  # one row per window

    t0 = time.time()
    for bstart in range(0, n, batch_size):
        bend = min(bstart + batch_size, n)
        q = np.asarray(qemb[bstart:bend], dtype=np.float32)
        raw_idx, raw_scores = index.search(q, CAND_K)  # both (bend-bstart, CAND_K), score-sorted descending
        raw_articles = chunk_article[raw_idx]
        batch_ids = np.full((bend - bstart, ARTICLE_K), -1, dtype=np.int32)
        batch_scores = np.full((bend - bstart, ARTICLE_K), -1e9, dtype=np.float32)
        for i in range(bend - bstart):
            # first hit per article is its best, since raw_idx is score-sorted; keep those in score order
            _, first = np.unique(raw_articles[i], return_index=True)
            keep = np.sort(first)[:ARTICLE_K]
            batch_ids[i, :len(keep)] = raw_idx[i, keep]
            batch_scores[i, :len(keep)] = raw_scores[i, keep]
        cand_ids_flat[bstart:bend] = batch_ids
        cand_scores_flat[bstart:bend] = batch_scores
        if (bstart // batch_size) % 10 == 0 or bend == n:
            rate = bend / max(time.time() - t0, 1e-6)
            print(f"search: {bend}/{n} windows ({rate:.0f}/s)")
    cand_ids.flush()
    cand_scores.flush()
    print(f"Wrote per-window article candidates for {R} rows to {PACK_DIR}")

# -----------------------------------------------------------------------------

def dedup_row(cand_ids_row, cand_scores_row, chunk_article):
    """
    Article-level dedup for one row's candidate lists.

    cand_ids_row/cand_scores_row: (num_windows, K), score-sorted descending per window,
    -1 id = no more candidates. chunk_article: chunk_id -> article_id lookup.

    Each window starts at its best candidate; whenever two windows share an article, the
    lower-scoring one advances to its next candidate. Repeats until no two windows share
    an article. Terminates because a window's candidate cursor only moves forward.
    Returns (chunk_ids, scores), each length num_windows (-1/-1e9 if a window ran out of
    candidates, which shouldn't happen once K is comfortably larger than num_windows).
    """
    num_windows, k = cand_ids_row.shape
    cursor = [0] * num_windows
    chunk = [int(cand_ids_row[w, 0]) for w in range(num_windows)]
    score = [float(cand_scores_row[w, 0]) for w in range(num_windows)]
    article = [int(chunk_article[c]) if c >= 0 else -1 for c in chunk]
    pairs = list(itertools.combinations(range(num_windows), 2))

    while True:
        conflict = next((p for p in pairs if article[p[0]] == article[p[1]] and article[p[0]] != -1), None)
        if conflict is None:
            break
        a, b = conflict
        loser = a if score[a] < score[b] else b
        cursor[loser] += 1
        j = cursor[loser]
        if j < k and cand_ids_row[loser, j] >= 0:
            chunk[loser] = int(cand_ids_row[loser, j])
            score[loser] = float(cand_scores_row[loser, j])
            article[loser] = int(chunk_article[chunk[loser]])
        else:
            chunk[loser], score[loser], article[loser] = -1, -1e9, -1  # ran out of candidates
    return chunk, score

def dedup_and_fill(split):
    rows = np.lib.format.open_memmap(packed_path(split), mode="r+")
    R = rows.shape[0]
    cand_ids = np.load(os.path.join(PACK_DIR, f"{split}_cand_ids.npy"), mmap_mode="r")
    cand_scores = np.load(os.path.join(PACK_DIR, f"{split}_cand_scores.npy"), mmap_mode="r")
    chunk_article = np.load(os.path.join(WIKI_DIR, "meta.npy"))[:, 0]
    wiki_tokens = np.load(os.path.join(WIKI_DIR, "tokens.npy"), mmap_mode="r")

    ref_ids = np.full((R, NUM_WINDOWS), -1, dtype=np.int32)
    ref_scores = np.full((R, NUM_WINDOWS), -1e9, dtype=np.float32)

    t0 = time.time()
    n_displaced = 0
    for row_idx in range(R):
        chunk, score = dedup_row(cand_ids[row_idx], cand_scores[row_idx], chunk_article)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join packed rows' windows to Wikipedia, with article-level dedup")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ew = sub.add_parser("embed-windows", help="Embed every window of every packed row")
    p_ew.add_argument("--split", choices=["train", "val"], required=True)
    p_ew.add_argument("--batch-size", type=int, default=256)
    p_ew.add_argument("--rank", type=int, default=0)
    p_ew.add_argument("--world-size", type=int, default=6)
    p_ew.add_argument("--gpu", type=int, default=None)

    p_search = sub.add_parser("search", help="Find each window's top distinct-article wiki chunk candidates")
    p_search.add_argument("--split", choices=["train", "val"], required=True)
    p_search.add_argument("--dtype", choices=["fp16", "int8"], default="fp16")
    p_search.add_argument("--gpus", type=int, nargs="+", default=None, help="GPUs to shard the wiki index across (default: all visible)")
    p_search.add_argument("--block-size", type=int, default=1_000_000)
    p_search.add_argument("--batch-size", type=int, default=4096)

    p_dedup = sub.add_parser("dedup", help="Resolve article-level conflicts and fill the packed rows' wiki columns")
    p_dedup.add_argument("--split", choices=["train", "val"], required=True)

    args = parser.parse_args()
    if args.command == "embed-windows":
        embed_windows(args.split, batch_size=args.batch_size, rank=args.rank, world_size=args.world_size, gpu=args.gpu)
    elif args.command == "search":
        search(args.split, wiki_dtype=args.dtype, gpus=args.gpus, block_size=args.block_size, batch_size=args.batch_size)
    elif args.command == "dedup":
        dedup_and_fill(args.split)
