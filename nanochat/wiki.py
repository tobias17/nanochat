"""
Wikipedia retrieval corpus for the encoder+decoder ("cheat sheet") experiment.

Builds a store of 256-token chunks from English Wikipedia plus sentence embeddings
for each chunk, so a packed pretraining row's 256-token windows (see nanochat.pack)
can each be joined to their nearest Wikipedia chunk by cosine similarity. Two
offline steps:

    python -m nanochat.wiki build   # tokenize + chunk all of Wikipedia
    python -m nanochat.wiki embed   # embed every chunk with mpnet

See scripts/wiki_retrieve.py (joins packed rows' windows to wiki chunks, with
article-level dedup) and scripts/wiki_view.py (inspect the results interactively).
"""
import os
import re
import json
import glob
import time
import argparse

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

# -----------------------------------------------------------------------------
# Paths

WIKI_DIR = os.path.join(get_base_dir(), "wiki")
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def _find_hf_wiki_dir():
    """ Locate the wikimedia/wikipedia 20231101.en snapshot cached by huggingface_hub. """
    pattern = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "datasets--wikimedia--wikipedia", "snapshots", "*", "20231101.en",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a cached wikimedia/wikipedia 20231101.en snapshot under {pattern}. "
            f"Download it first, e.g.: "
            f"python -c \""
            f"from huggingface_hub import snapshot_download; "
            f"snapshot_download('wikimedia/wikipedia', repo_type='dataset', allow_patterns='20231101.en/*')\""
        )
    return matches[-1]

def list_wiki_source_files():
    """ Raw Wikipedia parquet files: columns are id, url, title, text. """
    wiki_dir = _find_hf_wiki_dir()
    files = sorted(glob.glob(os.path.join(wiki_dir, "train-*.parquet")))
    assert files, f"No train-*.parquet files found in {wiki_dir}"
    return files

# -----------------------------------------------------------------------------
# Cleaning + chunking

# Drop everything from the first of these trailing section headers onward: they're
# link/citation lists that add tokens without adding retrievable content.
TRAILING_HEADERS = ["References", "External links", "See also", "Further reading", "Notes"]
_TRAILING_RE = re.compile(r"\n(?:" + "|".join(TRAILING_HEADERS) + r")\n")

def clean_article_text(text):
    m = _TRAILING_RE.search(text)
    return text[:m.start()] if m else text

def split_list_with_overlap(ids, chunk_len, target_overlap):
    """
    Split ids into chunks of exactly chunk_len tokens, evenly spaced so consecutive
    chunks overlap by about target_overlap tokens, with the last chunk end-aligned to
    len(ids) so it's never padded (except when a single chunk is chosen, which keeps
    the head and drops the short tail). Returns [] if there aren't enough tokens for
    even one full chunk (the article/window is dropped rather than padded).
    """
    assert 0 <= target_overlap < chunk_len
    n = len(ids)
    if n < chunk_len:
        return []
    k = round((n - chunk_len) / (chunk_len - target_overlap)) + 1
    if k == 1:
        return [ids[:chunk_len]]
    starts = [i * (n - chunk_len) // (k - 1) for i in range(k)]  # last start is exactly n - chunk_len
    return [ids[s:s + chunk_len] for s in starts]

# -----------------------------------------------------------------------------
# Build: raw Wikipedia -> token chunk store
#
# Streams through the source files, appending chunk tokens to a temporary raw file
# (rather than holding ~28M chunks in Python lists) so this scales to the full corpus.

def build(max_articles=-1, chunk_len=256, overlap=32, clean=True, prepend_title=True, out_dir=None):
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size <= 65535, f"vocab_size {vocab_size} does not fit in uint16 (got {vocab_size})"

    files = list_wiki_source_files()
    print(f"Found {len(files)} Wikipedia source files")

    out_dir = WIKI_DIR if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, ".tokens.raw")

    meta_parts = []     # per-file (n, 2) int32 arrays of (article_row, chunk_idx_within_article)
    all_titles, all_urls = [], []

    # total row count across all files, for a percent/ETA against the whole corpus (cheap: metadata only, no data read)
    total_rows = sum(pq.read_metadata(p).num_rows for p in files)
    if max_articles >= 0:
        total_rows = min(total_rows, max_articles)

    max_title_len = chunk_len // 2  # a (pathologically) long title never takes more than half a chunk
    assert overlap < chunk_len - max_title_len
    article_row = 0
    n_chunks = 0
    t0 = time.time()
    with open(raw_path, "wb") as tf:
        for fi, path in enumerate(files):
            tf0 = time.time()
            table = pq.read_table(path, columns=["title", "url", "text"])
            if max_articles >= 0:
                table = table.slice(0, max_articles - article_row)
            n_rows = table.num_rows
            print(f"[{fi + 1}/{len(files)}] {os.path.basename(path)}: read {n_rows} rows in {time.time() - tf0:.0f}s", flush=True)

            t1 = time.time()
            titles = table.column("title").to_pylist()
            urls = table.column("url").to_pylist()
            texts = table.column("text").to_pylist()
            print(f"[{fi + 1}/{len(files)}]   to_pylist: {time.time() - t1:.0f}s", flush=True)

            t1 = time.time()
            bodies = [clean_article_text(t) if clean else t for t in texts]
            print(f"[{fi + 1}/{len(files)}]   clean: {time.time() - t1:.0f}s", flush=True)

            t1 = time.time()
            body_id_lists = tokenizer.encode(bodies, num_threads=8)
            print(f"[{fi + 1}/{len(files)}]   encode bodies: {time.time() - t1:.0f}s", flush=True)

            t1 = time.time()
            title_id_lists = tokenizer.encode([t + "\n\n" for t in titles], num_threads=8) if prepend_title else None
            print(f"[{fi + 1}/{len(files)}]   encode titles: {time.time() - t1:.0f}s", flush=True)

            t1 = time.time()
            batch_chunks, batch_meta = [], []
            for i, body_ids in enumerate(body_id_lists):
                if prepend_title:
                    title_ids = title_id_lists[i][:max_title_len]
                    budget = chunk_len - len(title_ids)
                    chunks = [title_ids + c for c in split_list_with_overlap(body_ids, budget, overlap)]
                else:
                    chunks = split_list_with_overlap(body_ids, chunk_len, overlap)
                for ci, c in enumerate(chunks):
                    assert len(c) == chunk_len
                    batch_meta.append((article_row, ci))
                    batch_chunks.append(c)
                article_row += 1
            all_titles.extend(titles)
            all_urls.extend(urls)
            print(f"[{fi + 1}/{len(files)}]   chunk: {time.time() - t1:.0f}s", flush=True)

            t1 = time.time()
            if batch_chunks:
                np.asarray(batch_chunks, dtype=np.uint16).tofile(tf)
                n_chunks += len(batch_chunks)
            meta_parts.append(np.asarray(batch_meta, dtype=np.int32).reshape(-1, 2))
            print(f"[{fi + 1}/{len(files)}]   write: {time.time() - t1:.0f}s", flush=True)

            elapsed = time.time() - t0
            pct = 100 * article_row / max(total_rows, 1)
            rate = article_row / max(elapsed, 1e-6)
            eta = (total_rows - article_row) / max(rate, 1e-6)
            print(f"[{fi + 1}/{len(files)}] done: {article_row}/{total_rows} articles ({pct:.1f}%), "
                  f"{n_chunks} chunks so far, {elapsed:.0f}s elapsed, ETA {eta / 60:.1f}m", flush=True)
            if article_row == max_articles:
                break

    tokens = np.memmap(raw_path, dtype=np.uint16, mode="r", shape=(n_chunks, chunk_len))
    np.save(os.path.join(out_dir, "tokens.npy"), tokens)
    del tokens
    os.remove(raw_path)

    np.save(os.path.join(out_dir, "meta.npy"), np.concatenate(meta_parts))

    articles_table = pa.table({
        "row": np.arange(article_row, dtype=np.int32),
        "title": all_titles,
        "url": all_urls,
    })
    pq.write_table(articles_table, os.path.join(out_dir, "articles.parquet"))

    config = dict(
        chunk_len=chunk_len, overlap=overlap, clean=clean, prepend_title=prepend_title,
        max_articles=max_articles, num_articles=article_row, num_chunks=n_chunks,
        vocab_size=vocab_size, embed_model=EMBED_MODEL_NAME, source="wikimedia/wikipedia 20231101.en",
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote {n_chunks} chunks from {article_row} articles to {out_dir}")

# -----------------------------------------------------------------------------
# Embed: token rows -> mpnet sentence embeddings, sharded across `world_size` ranks
#
# Embeddings are L2-normalized, so cosine similarity is a plain dot product. The
# "int8" dtype quantizes each unit-norm component to a signed byte (scale 127,
# fixed since |component| <= 1 for a unit vector) -- half the size of fp16, at a
# small cosine-similarity error, so a resident index is cheaper to hold on GPU.

def _shard_bounds(n, rank, world_size):
    """ Contiguous, near-equal row range [start, end) for this rank (handles n % world_size != 0). """
    block, rem = divmod(n, world_size)
    start = rank * block + min(rank, rem)
    end = start + block + (1 if rank < rem else 0)
    return start, end

def embed_id_lists(id_lists, tokenizer, model, dtype="fp16"):
    """ Decode token-id lists back to text and embed them. Returns an (N, dim) array in `dtype`. """
    texts = [tokenizer.decode(ids) for ids in id_lists]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    if dtype == "int8":
        return np.clip(np.round(vecs * 127.0), -127, 127).astype(np.int8)
    return vecs.astype(np.float16)

def embed_rows_to_memmap(row_ids_fn, n, out_path, model, tokenizer, batch_size=256, dtype="fp16",
                          rank=0, world_size=1, log_prefix=""):
    """
    row_ids_fn(i) -> list[int] token ids for row i (already whatever's meant to be embedded,
    e.g. BOS stripped). Embeds rows [start, end) of this rank's shard into a memmap at
    out_path shared by all ranks, which may be running concurrently.
    """
    start, end = _shard_bounds(n, rank, world_size)
    emb_dim = model.get_sentence_embedding_dimension()
    emb_dtype = np.int8 if dtype == "int8" else np.float16
    if not os.path.exists(out_path):
        # create under a private name, then hard-link into place: os.link never overwrites, so if
        # several ranks race here exactly one file wins and no rank truncates another's writes
        tmp_path = f"{out_path}.rank{rank}.tmp"
        np.lib.format.open_memmap(tmp_path, mode="w+", dtype=emb_dtype, shape=(n, emb_dim))  # header + sparse zeroed body
        try:
            os.link(tmp_path, out_path)
        except FileExistsError:
            pass
        os.remove(tmp_path)
    emb = np.lib.format.open_memmap(out_path, mode="r+")
    assert emb.shape == (n, emb_dim) and emb.dtype == emb_dtype, (
        f"{out_path} is {emb.shape} {emb.dtype}, expected {(n, emb_dim)} {np.dtype(emb_dtype)} -- stale from an older run? delete it"
    )

    t0 = time.time()
    n_resumed = 0
    for bstart in range(start, end, batch_size):
        bend = min(bstart + batch_size, end)
        if emb[bend - 1].any():
            # Resume: each rank writes its shard front to back, and a unit-norm embedding is never all
            # zeros, so a nonzero last row means an earlier (interrupted) run already did this batch.
            # (Only valid if that run used the same world_size, i.e. the same shard boundaries.)
            n_resumed += bend - bstart
            continue
        id_lists = [row_ids_fn(i) for i in range(bstart, bend)]
        emb[bstart:bend] = embed_id_lists(id_lists, tokenizer, model, dtype)
        done, total = bend - start, end - start
        if (bstart // batch_size) % 20 == 0 or bend == end:
            rate = (done - n_resumed) / max(time.time() - t0, 1e-6)
            eta = (total - done) / max(rate, 1e-6)
            print(f"{log_prefix}rank {rank}: {done}/{total} rows ({rate:.0f}/s, eta {eta / 60:.1f}m)")
    if n_resumed:
        print(f"{log_prefix}rank {rank}: skipped {n_resumed} rows already embedded by an earlier run")
    emb.flush()
    return start, end

def embed(batch_size=256, dtype="fp16", rank=0, world_size=6, gpu=None, wiki_dir=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is already cached; don't hit the network to revalidate it
    from sentence_transformers import SentenceTransformer

    assert dtype in ("fp16", "int8")
    wiki_dir = WIKI_DIR if wiki_dir is None else wiki_dir
    gpu = rank if gpu is None else gpu  # which physical CUDA device to run this rank's shard on
    tokenizer = get_tokenizer()
    tokens = np.load(os.path.join(wiki_dir, "tokens.npy"), mmap_mode="r")
    n = tokens.shape[0]

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    emb_name = "emb_int8.npy" if dtype == "int8" else "emb.npy"
    emb_path = os.path.join(wiki_dir, emb_name)
    start, end = embed_rows_to_memmap(
        lambda i: tokens[i].tolist(), n, emb_path, model, tokenizer,
        batch_size=batch_size, dtype=dtype, rank=rank, world_size=world_size, log_prefix=f"wiki cuda:{gpu} ",
    )
    print(f"rank {rank} (cuda:{gpu}): done, wrote rows [{start}:{end}) of {emb_path}")

# -----------------------------------------------------------------------------
# Search: brute-force top-k over the wiki index, sharded across one or more GPUs
#
# Each GPU holds a contiguous slice of the index resident in its stored dtype
# (fp16 or int8) and searches only its own slice; per-shard top-k results are then
# merged on CPU. Each shard's search only upcasts one block at a time to fp16 for
# the matmul, so peak memory per GPU is (resident shard) + (one block in fp16).

class WikiIndex:
    def __init__(self, dtype="fp16", gpus=None, block_size=1_000_000, wiki_dir=None):
        assert dtype in ("fp16", "int8")
        self.dtype = dtype
        self.block_size = block_size
        wiki_dir = WIKI_DIR if wiki_dir is None else wiki_dir
        if gpus is None:  # default: shard across every visible GPU (the full index doesn't fit on one)
            gpus = list(range(torch.cuda.device_count())) or ["cpu"]
        self.devices = [f"cuda:{g}" if isinstance(g, int) else g for g in gpus]

        emb_path = os.path.join(wiki_dir, "emb_int8.npy" if dtype == "int8" else "emb.npy")
        assert os.path.exists(emb_path), (
            f"Missing {emb_path}. Run: python -m nanochat.wiki embed --dtype {dtype}"
        )
        emb_np = np.load(emb_path, mmap_mode="r")
        self.n, self.dim = emb_np.shape
        np_dtype = torch.int8 if dtype == "int8" else torch.float16

        self.shards = []  # (device, row_start, resident_tensor)
        shard_n = -(-self.n // len(self.devices))  # ceil div
        for si, device in enumerate(self.devices):
            row_start, row_end = si * shard_n, min((si + 1) * shard_n, self.n)
            if row_start >= row_end:
                continue
            resident = torch.from_numpy(np.array(emb_np[row_start:row_end])).to(device, dtype=np_dtype)
            self.shards.append((device, row_start, resident))

    def search(self, query_vecs, k):
        """ query_vecs: (Q, dim) float array, L2-normalized. Returns (idx, score), both (Q, k). """
        q_np = query_vecs.detach().cpu().numpy() if isinstance(query_vecs, torch.Tensor) else np.asarray(query_vecs, dtype=np.float32)
        Q = q_np.shape[0]
        shard_scores, shard_idx = [], []
        for device, row_start, resident in self.shards:
            q = torch.from_numpy(q_np).to(device, dtype=torch.float16)
            n_rows = resident.shape[0]
            best_scores = torch.full((Q, k), -1e9, device=device, dtype=torch.float32)
            best_idx = torch.zeros((Q, k), dtype=torch.long, device=device)
            for bstart in range(0, n_rows, self.block_size):
                bend = min(bstart + self.block_size, n_rows)
                block = resident[bstart:bend]
                block = (block.to(torch.float16) / 127.0) if self.dtype == "int8" else block
                scores = (q @ block.T).float()
                kk = min(k, scores.shape[1])
                vals, idxs = torch.topk(scores, k=kk, dim=1)
                idxs = idxs + row_start + bstart
                cat_scores = torch.cat([best_scores, vals], dim=1)
                cat_idx = torch.cat([best_idx, idxs], dim=1)
                top_vals, top_pos = torch.topk(cat_scores, k=k, dim=1)
                best_scores = top_vals
                best_idx = torch.gather(cat_idx, 1, top_pos)
            shard_scores.append(best_scores.cpu())
            shard_idx.append(best_idx.cpu())
        all_scores = torch.cat(shard_scores, dim=1)
        all_idx = torch.cat(shard_idx, dim=1)
        top_vals, top_pos = torch.topk(all_scores, k=k, dim=1)
        top_idx = torch.gather(all_idx, 1, top_pos)
        return top_idx.numpy(), top_vals.numpy()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Wikipedia retrieval corpus")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Tokenize + chunk raw Wikipedia into the token store")
    p_build.add_argument("--max-articles", type=int, default=-1, help="Cap articles for a quick test run (-1 = all)")
    p_build.add_argument("--chunk-len", type=int, default=256, help="Tokens per chunk (matches the packed row window size)")
    p_build.add_argument("--overlap", type=int, default=32, help="Target overlapping tokens between consecutive chunks of the same article")
    p_build.add_argument("--no-clean", action="store_true", help="Keep References/External links/etc. instead of stripping them")
    p_build.add_argument("--no-title", action="store_true", help="Don't prepend the article title to each chunk")

    p_embed = sub.add_parser("embed", help="Embed every chunk with all-mpnet-base-v2")
    p_embed.add_argument("--batch-size", type=int, default=256)
    p_embed.add_argument("--dtype", choices=["fp16", "int8"], default="fp16", help="int8 halves the index's memory footprint at a small recall cost")
    p_embed.add_argument("--rank", type=int, default=0, help="Which shard of the corpus this process embeds")
    p_embed.add_argument("--world-size", type=int, default=6, help="How many shards to split the corpus into")
    p_embed.add_argument("--gpu", type=int, default=None, help="CUDA device to run on (default: same as --rank)")

    args = parser.parse_args()
    if args.command == "build":
        build(max_articles=args.max_articles, chunk_len=args.chunk_len, overlap=args.overlap,
              clean=not args.no_clean, prepend_title=not args.no_title)
    elif args.command == "embed":
        embed(batch_size=args.batch_size, dtype=args.dtype, rank=args.rank, world_size=args.world_size, gpu=args.gpu)
