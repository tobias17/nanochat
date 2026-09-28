"""
Interactively inspect the packed-row <-> Wikipedia join: for a packed pretraining
row, show its 8 document windows next to the Wikipedia chunk retrieved for each.

    python -m scripts.wiki_view --split train
    python -m scripts.wiki_view --split train --start 100
    python -m scripts.wiki_view --split train --stats
    python -m scripts.wiki_view --query "some text"
    python -m scripts.wiki_view --task gsm8k_main_test [--stats]   # SFT/eval conversations' refs (see wiki_retrieve tasks)

Controls: Enter = next row, q = quit, a<N> = jump to row N, r = random.
"""
import os
import sys
import shutil
import argparse
import random
import textwrap
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer
from nanochat.pack import PACK_DIR, DOC_WIDTH, NUM_WINDOWS, WINDOW_LEN, packed_path
from nanochat.wiki import WIKI_DIR, EMBED_MODEL_NAME, WikiIndex
from tasks.common import REFS_DIR

# -----------------------------------------------------------------------------
# Terminal styling (disabled automatically when stdout isn't a real terminal,
# e.g. piped to a file, so redirected output stays plain text)

_COLOR = sys.stdout.isatty()
def _c(code):
    return code if _COLOR else ""

RESET = _c("\033[0m")
BOLD = _c("\033[1m")
DIM = _c("\033[2m")
RED = _c("\033[91m")
BOLD_RED = _c("\033[1;91m")
GREEN = _c("\033[92m")
YELLOW = _c("\033[93m")
CYAN = _c("\033[96m")
BOLD_CYAN = _c("\033[1;96m")

BOUNDARY_MARK = f"{BOLD_RED}[BOUNDARY]{RESET}"

def score_color(score):
    """ Rough at-a-glance quality signal: green = strong match, yellow = mediocre, red = weak. """
    if score >= 0.5:
        return GREEN
    if score >= 0.3:
        return YELLOW
    return RED

def _terminal_width(margin=2):
    return max(40, shutil.get_terminal_size(fallback=(100, 24)).columns - margin)

def wrap(text, width=None):
    width = _terminal_width() if width is None else width
    return "\n".join(textwrap.wrap(text, width=width) or [""])

def split_on_bos(ids, bos_id):
    """ Split a token id list into per-document segments at BOS boundaries. """
    segments, cur = [], []
    for t in ids:
        if t == bos_id:
            if cur:
                segments.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        segments.append(cur)
    return segments

def print_wrapped_with_boundaries(tokenizer, ids, bos_id):
    """ Decode+print ids, with an inline marker at every document boundary. """
    segments = split_on_bos(ids, bos_id)
    text = f" {BOUNDARY_MARK} ".join(tokenizer.decode(seg) for seg in segments)
    print(wrap(text))

class WikiLookup:
    def __init__(self):
        self.tokens = np.load(os.path.join(WIKI_DIR, "tokens.npy"), mmap_mode="r")
        self.meta = np.load(os.path.join(WIKI_DIR, "meta.npy"))
        articles = pq.read_table(os.path.join(WIKI_DIR, "articles.parquet"))
        self.titles = articles.column("title").to_pylist()
        self.tokenizer = get_tokenizer()

    def describe(self, chunk_id):
        article_row, chunk_idx = self.meta[chunk_id].tolist()
        title = self.titles[article_row]
        text = self.tokenizer.decode(self.tokens[chunk_id].tolist())
        return title, chunk_idx, text

class RowViewer:
    def __init__(self, split):
        self.split = split
        self.rows = np.load(packed_path(split), mmap_mode="r")
        ref_ids_path = os.path.join(PACK_DIR, f"{split}_ref_ids.npy")
        assert os.path.exists(ref_ids_path), (
            f"No reference ids for split={split!r} at {ref_ids_path}. Run the wiki_retrieve pipeline first:\n"
            f"  python -m scripts.wiki_retrieve embed-windows --split {split}\n"
            f"  python -m scripts.wiki_retrieve search-shard --split {split}\n"
            f"  python -m scripts.wiki_retrieve search-merge --split {split}\n"
            f"  python -m scripts.wiki_retrieve dedup --split {split}"
        )
        self.ref_ids = np.load(ref_ids_path)
        self.ref_scores = np.load(os.path.join(PACK_DIR, f"{split}_ref_scores.npy"))
        self.tokenizer = get_tokenizer()
        self.bos_id = self.tokenizer.get_bos_token_id()
        self.lookup = WikiLookup()

    def num_rows(self):
        return self.rows.shape[0]

    def doc_ids(self, row_idx):
        return self.rows[row_idx, :DOC_WIDTH - 1].tolist()  # the 2048 input tokens (drop the +1 target token)

    def window_ids(self, row_idx, w):
        return self.rows[row_idx, w * WINDOW_LEN:(w + 1) * WINDOW_LEN].tolist()

def show_row(row_idx, viewer):
    ids = viewer.doc_ids(row_idx)
    width = _terminal_width()
    print(f"{BOLD_CYAN}{'━' * width}{RESET}")
    print(f"{BOLD}ROW {row_idx}{RESET}  {DIM}({len(ids)} tokens){RESET}")
    print(f"{DIM}{'─' * width}{RESET}")
    print_wrapped_with_boundaries(viewer.tokenizer, ids, viewer.bos_id)
    print()
    for w in range(NUM_WINDOWS):
        window = viewer.window_ids(row_idx, w)
        spans_boundary = viewer.bos_id in window[1:]  # a boundary after position 0 means 2 docs in this window
        ref_id = int(viewer.ref_ids[row_idx, w])
        score = float(viewer.ref_scores[row_idx, w])
        boundary_tag = f"  {BOLD_RED}[spans document boundary]{RESET}" if spans_boundary else ""
        print(f"{BOLD}── window {w}{RESET} {DIM}(tokens {w * WINDOW_LEN}-{(w + 1) * WINDOW_LEN}){RESET}{boundary_tag}")
        print_wrapped_with_boundaries(viewer.tokenizer, window, viewer.bos_id)
        if ref_id >= 0:
            title, chunk_idx, chunk_text = viewer.lookup.describe(ref_id)
            sc = score_color(score)
            print(f"  {sc}{BOLD}→ score={score:.4f}{RESET}  {BOLD_CYAN}{title}{RESET} {DIM}(chunk {chunk_idx}){RESET}")
            print("  " + wrap(chunk_text).replace("\n", "\n  "))
        else:
            print(f"  {BOLD_RED}→ no reference (ran out of distinct-article candidates){RESET}")
        print()

def interactive(viewer, args):
    n = viewer.num_rows()
    order = list(range(args.start, n))
    if args.random:
        random.shuffle(order)
    pos = 0
    while 0 <= pos < len(order):
        row_idx = order[pos]
        top_score = float(viewer.ref_scores[row_idx].max())
        if args.min_score is not None and top_score < args.min_score:
            pos += 1
            continue
        show_row(row_idx, viewer)
        cmd = input(f"{DIM}[Enter=next, q=quit, a<N>=jump, r=random]{RESET} > ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "r":
            order.insert(pos + 1, random.randrange(n))  # show it next
        elif cmd.startswith("a"):
            if cmd[1:].isdigit() and int(cmd[1:]) < n:
                order.insert(pos + 1, int(cmd[1:]))
            else:
                print(f"Could not parse a row index in [0, {n}) from {cmd!r}")
        pos += 1

def print_stats(split):
    ref_scores = np.load(os.path.join(PACK_DIR, f"{split}_ref_scores.npy"))
    ref_ids = np.load(os.path.join(PACK_DIR, f"{split}_ref_ids.npy"))
    cand_path = os.path.join(PACK_DIR, f"{split}_cand_scores.npy")
    R = ref_scores.shape[0]
    print(f"{BOLD_CYAN}{'━' * _terminal_width()}{RESET}")
    print(f"{BOLD}rows{RESET}: {R}   {BOLD}windows/row{RESET}: {NUM_WINDOWS}")

    flat = ref_scores.reshape(-1)
    pct = np.percentile(flat, [1, 10, 25, 50, 75, 90, 99])
    print(f"{BOLD}post-dedup scores{RESET}: p1={pct[0]:.3f} p10={pct[1]:.3f} p25={pct[2]:.3f} "
          f"{BOLD}p50={pct[3]:.3f}{RESET} p75={pct[4]:.3f} p90={pct[5]:.3f} p99={pct[6]:.3f}")
    frac_none = float(np.mean(ref_ids < 0))
    c = GREEN if frac_none == 0 else BOLD_RED
    print(f"{BOLD}fraction with no reference{RESET} (ran out of candidates): {c}{frac_none:.4f}{RESET}")

    if os.path.exists(cand_path):
        cand_scores = np.load(cand_path, mmap_mode="r")
        top1 = cand_scores[:, :, 0].reshape(-1)  # pre-dedup best score per window
        displaced = ref_scores.reshape(-1) < (top1 - 1e-6)
        frac_disp = float(np.mean(displaced))
        c = GREEN if frac_disp < 0.1 else (YELLOW if frac_disp < 0.3 else BOLD_RED)
        print(f"{BOLD}fraction of windows displaced by dedup{RESET}: {c}{frac_disp:.4f}{RESET}")
        if displaced.any():
            cost = top1[displaced] - ref_scores.reshape(-1)[displaced]
            print(f"{BOLD}score cost of displacement{RESET}: mean={cost.mean():.4f} p50={np.percentile(cost,50):.4f} p90={np.percentile(cost,90):.4f}")

    meta = np.load(os.path.join(WIKI_DIR, "meta.npy"))
    articles = pq.read_table(os.path.join(WIKI_DIR, "articles.parquet")).column("title").to_pylist()
    valid = ref_ids[ref_ids >= 0]
    article_ids = meta[valid, 0]
    counts = Counter(article_ids.tolist())
    print(f"{DIM}{'─' * _terminal_width()}{RESET}")
    print(f"{BOLD}most-referenced articles{RESET} (title: count):")
    for art, cnt in counts.most_common(10):
        print(f"  {CYAN}{articles[art]!r}{RESET}: {cnt}")

def query_mode(query_text, args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is already cached; don't hit the network to revalidate it
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_NAME, device=f"cuda:{args.gpu}")
    index = WikiIndex(dtype=args.dtype)  # sharded across all visible GPUs; the full index doesn't fit on one
    qvec = model.encode([query_text], normalize_embeddings=True)
    ids, scores = index.search(qvec, args.k)
    lookup = WikiLookup()
    width = _terminal_width()
    print(f"{BOLD_CYAN}{'━' * width}{RESET}")
    print(f"{BOLD}QUERY{RESET}  {DIM}({len(query_text)} chars){RESET}")
    print(f"{DIM}{'─' * width}{RESET}")
    print(wrap(query_text))
    print()
    for rank in range(min(args.k, len(ids[0]))):
        title, chunk_idx, chunk_text = lookup.describe(int(ids[0][rank]))
        sc = score_color(float(scores[0][rank]))
        print(f"{BOLD}[{rank + 1}]{RESET} {sc}{BOLD}score={scores[0][rank]:.4f}{RESET}  {BOLD_CYAN}{title}{RESET} {DIM}(chunk {chunk_idx}){RESET}")
        print(wrap(chunk_text))
        print()

def task_mode(key, args):
    """ Page through a task's queries, each with its precomputed refs. """
    from scripts.wiki_retrieve import _ref_tasks
    task = _ref_tasks()[key]()
    ids = np.load(os.path.join(REFS_DIR, f"{key}_ids.npy"))
    scores = np.load(os.path.join(REFS_DIR, f"{key}_scores.npy"))
    if args.stats:
        valid = scores[scores > -1e8]
        print(f"{key}: {len(ids)} queries x {ids.shape[1]} refs, {int((ids < 0).any(axis=1).sum())} queries with missing refs")
        print("score percentiles p1/p5/p25/p50/p75/p95:", np.percentile(valid, [1, 5, 25, 50, 75, 95]).round(3).tolist())
        print("best-ref score p5/p50/p95:", np.percentile(scores[:, 0], [5, 50, 95]).round(3).tolist())
        return
    lookup = WikiLookup()
    n = len(ids)
    order = list(range(args.start, n))
    if args.random:
        random.shuffle(order)
    pos = 0
    while 0 <= pos < len(order):
        i = order[pos]
        print(f"{BOLD_CYAN}=== {key}[{i}] ==={RESET}")
        print(wrap(task.retrieval_query(i)[:1500]))
        print()
        for chunk_id, score in zip(ids[i].tolist(), scores[i].tolist()):
            if chunk_id < 0:
                print(f"  {BOLD_RED}→ missing ref{RESET}")
                continue
            title, chunk_idx, text = lookup.describe(chunk_id)
            print(f"  {BOLD}{title}{RESET} #{chunk_idx}  {score_color(score)}{score:.3f}{RESET}")
            print(f"  {DIM}{wrap(text[:300], _terminal_width(6)).replace(chr(10), chr(10) + '  ')}{RESET}")
        print()
        cmd = input(f"{DIM}[Enter=next, q=quit, a<N>=jump, r=random]{RESET} > ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "r":
            order.insert(pos + 1, random.randrange(n))
        elif cmd.startswith("a") and cmd[1:].isdigit() and int(cmd[1:]) < n:
            order.insert(pos + 1, int(cmd[1:]))
        pos += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the packed-row <-> Wikipedia retrieval join")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--start", type=int, default=0, help="Row index to start at")
    parser.add_argument("--random", action="store_true", help="Walk rows in random order")
    parser.add_argument("--min-score", type=float, default=None, help="Only show rows whose best window score is >= this")
    parser.add_argument("--stats", action="store_true", help="Print score/coverage statistics instead of paging interactively")
    parser.add_argument("--query", type=str, default=None, help="Embed this text ad hoc and show its nearest wiki chunks")
    parser.add_argument("--k", type=int, default=4, help="Neighbours to show (for --query)")
    parser.add_argument("--dtype", choices=["fp16", "int8"], default="fp16", help="Which wiki embedding index to load (for --query)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to run the embedding model on (for --query)")
    parser.add_argument("--task", type=str, default=None, help="Show a task's precomputed refs by ref_key, e.g. smoltalk_train")
    args = parser.parse_args()

    if args.task is not None:
        task_mode(args.task, args)
    elif args.query is not None:
        query_mode(args.query, args)
    elif args.stats:
        print_stats(args.split)
    else:
        viewer = RowViewer(args.split)
        interactive(viewer, args)
