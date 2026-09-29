"""
Tests for offline row packing (nanochat.dataloader.BestFitPacker, shared by the live
training loader and nanochat.pack), wiki chunking (nanochat.wiki.split_list_with_overlap),
retrieval dedup (scripts.wiki_retrieve.dedup_row), the packed-row training loader
(nanochat.dataloader.packed_data_loader_with_state) and embedding resume (nanochat.wiki).

Hermetic: trains a tiny throwaway tokenizer in-process, no dependency on ~/.cache/nanochat.

python -m pytest tests/test_packing.py -v
"""
import numpy as np
import pytest
import torch

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
import nanochat.pack
from nanochat.pack import DOC_WIDTH, ROW_WIDTH, NUM_WINDOWS, WINDOW_LEN
from nanochat.dataloader import BestFitPacker, packed_data_loader_with_state
from nanochat.wiki import split_list_with_overlap, embed_rows_to_memmap
from scripts.wiki_retrieve import dedup_row, DISTINCT_K

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello world, hello tokenizer, hello hello hello",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
] * 8

@pytest.fixture(scope="module")
def tokenizer():
    vocab_size = 256 + len(SPECIAL_TOKENS) + 35
    return RustBPETokenizer.train_from_iterator(iter(CORPUS), vocab_size)

def _fake_batches(doc_texts, batch_size=4):
    """ Matches nanochat.dataloader._document_batches's yield shape, for a fixed doc list. """
    epoch = 1
    while True:
        for i in range(0, len(doc_texts), batch_size):
            yield doc_texts[i:i + batch_size], (0, 0, epoch)
        epoch += 1

# -----------------------------------------------------------------------------
# BestFitPacker

def test_pack_row_exact_capacity_no_padding(tokenizer):
    docs = [("word " * n).strip() for n in [3, 5, 40, 1, 8, 2, 50, 4, 12, 25]]
    packer = BestFitPacker(_fake_batches(docs, batch_size=3), tokenizer, buffer_size=5)
    for row_capacity in [15, 40, 100]:
        row = packer.pack_row(row_capacity)
        assert len(row) == row_capacity

def test_pack_row_starts_with_bos(tokenizer):
    docs = [("word " * n).strip() for n in [5, 10, 15, 20]]
    packer = BestFitPacker(_fake_batches(docs, batch_size=2), tokenizer, buffer_size=3)
    row = packer.pack_row(30)
    assert row[0] == tokenizer.get_bos_token_id()

def test_pack_row_best_fit_prefers_exact_match_over_cropping(tokenizer):
    # Craft two docs of known token length: one exactly fills the remaining capacity.
    bos = tokenizer.get_bos_token_id()
    doc_a = "word " * 3   # short
    doc_b = "word " * 10  # long enough that if it fit exactly, best-fit should pick it whole
    ids_a = tokenizer.encode(doc_a, prepend=bos)
    ids_b = tokenizer.encode(doc_b, prepend=bos)
    row_capacity = len(ids_a) + len(ids_b)  # exactly fits both docs with nothing left over
    packer = BestFitPacker(_fake_batches([doc_a, doc_b], batch_size=2), tokenizer, buffer_size=2)
    row = packer.pack_row(row_capacity)
    # both docs fit exactly, so the row is not cropped: largest-first, it's doc_b then doc_a, whole
    assert row == ids_b + ids_a

def test_pack_row_state_dict_tracks_progress(tokenizer):
    docs = [("word " * n).strip() for n in range(1, 20)]
    packer = BestFitPacker(_fake_batches(docs, batch_size=3), tokenizer, buffer_size=4)
    assert packer.state_dict["epoch"] == 1
    for _ in range(10):  # 19 docs are ~200 tokens total, so 10 rows of 50 must cycle past the first epoch
        packer.pack_row(50)
    assert packer.state_dict["epoch"] > 1

# -----------------------------------------------------------------------------
# wiki chunking

def test_split_list_with_overlap_exact_length_and_end_alignment():
    ids = list(range(600))  # long enough that the formula picks >1 chunk at overlap=32
    chunks = split_list_with_overlap(ids, chunk_len=256, target_overlap=32)
    assert all(len(c) == 256 for c in chunks)
    assert chunks[0] == ids[:256]
    assert chunks[-1] == ids[-256:]  # last chunk end-aligned, never padded

def test_split_list_with_overlap_drops_short_input():
    assert split_list_with_overlap(list(range(100)), chunk_len=256, target_overlap=32) == []

def test_split_list_with_overlap_exact_length_no_split_needed():
    ids = list(range(256))
    assert split_list_with_overlap(ids, chunk_len=256, target_overlap=32) == [ids]

def test_split_list_with_overlap_consecutive_chunks_overlap():
    ids = list(range(1000))
    chunks = split_list_with_overlap(ids, chunk_len=256, target_overlap=32)
    for c1, c2 in zip(chunks, chunks[1:]):
        assert set(c1) & set(c2), "consecutive chunks of a long article should overlap"

# -----------------------------------------------------------------------------
# retrieval dedup

def test_dedup_row_no_conflicts_keeps_top1():
    cand_ids = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
    cand_scores = np.array([[0.9, 0.8, 0.7], [0.85, 0.5, 0.4], [0.6, 0.3, 0.2]])
    content_ids = {c: c for c in cand_ids.flatten().tolist()}
    chunk, score = dedup_row(cand_ids, cand_scores, content_ids)
    assert chunk == [10, 20, 30]

def test_dedup_row_resolves_conflict_by_score_and_terminates():
    # windows 0 and 1 both want chunk 10 at rank 0; window 1 has the higher score and should win
    cand_ids = np.array([[10, 13], [10, 14]])
    cand_scores = np.array([[0.5, 0.1], [0.9, 0.2]])
    content_ids = {10: 10, 13: 13, 14: 14}
    chunk, score = dedup_row(cand_ids, cand_scores, content_ids)
    assert chunk == [13, 10]  # window 1 keeps its top choice, window 0 was displaced to its next candidate
    assert score == [0.1, 0.9]

def test_dedup_row_identical_texts_conflict_but_same_article_does_not():
    # chunk 11 is a byte-identical copy of chunk 10 (content id 10); chunk 12 is a different text,
    # e.g. the next chunk of the same article, which is allowed alongside 10
    cand_ids = np.array([[10, 20], [11, 21], [12, 22]])
    cand_scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    content_ids = {10: 10, 11: 10, 12: 12, 20: 20, 21: 21, 22: 22}
    chunk, score = dedup_row(cand_ids, cand_scores, content_ids)
    assert chunk == [10, 21, 12]

def test_dedup_row_no_duplicate_texts_with_heavy_overlap():
    num_windows = 8
    rng = np.random.default_rng(0)
    cand_ids = np.arange(num_windows * DISTINCT_K).reshape(num_windows, DISTINCT_K)
    cand_scores = -np.sort(-rng.random((num_windows, DISTINCT_K)), axis=1)  # descending, like real candidates
    # every window's 32 candidates cycle through 16 distinct texts (32 % 16 == 0),
    # so conflicts are frequent but always resolvable
    content_ids = {int(c): int(c) % 16 for c in cand_ids.flatten()}
    chunk, score = dedup_row(cand_ids, cand_scores, content_ids)
    assert all(c >= 0 for c in chunk)  # never runs out of candidates in this regime
    assigned = [content_ids[c] for c in chunk]
    assert len(assigned) == len(set(assigned))

# -----------------------------------------------------------------------------
# packed-row training loader

WORLD_SIZE = 3
STREAM_LEN = 10

@pytest.fixture
def packed_rows(tmp_path, monkeypatch):
    """ A fake packed file laid out like nanochat.pack writes it: file row i*WORLD_SIZE+r is stream r's i-th row.
    Every token of a row is its file row index (+1 in the ref columns, to tell them apart from the doc). """
    rows = np.repeat(np.arange(STREAM_LEN * WORLD_SIZE, dtype=np.uint16)[:, None], ROW_WIDTH, axis=1)
    rows[:, DOC_WIDTH:] += 1
    path = tmp_path / "train.npy"
    np.save(path, rows)
    monkeypatch.setattr(nanochat.pack, "packed_path", lambda split: str(path))
    return rows

def _load(monkeypatch, rank, B, n_batches, resume_state_dict=None):
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(WORLD_SIZE))
    loader = packed_data_loader_with_state(B, DOC_WIDTH - 1, "train", device="cpu", resume_state_dict=resume_state_dict)
    out = []
    for _ in range(n_batches):
        x, y, refs, state = next(loader)
        out.append((x.clone(), y.clone(), refs.clone(), state))
    return out

def test_packed_loader_walks_own_stream_in_order(packed_rows, monkeypatch):
    B = 2
    for rank in range(WORLD_SIZE):
        batches = _load(monkeypatch, rank, B, n_batches=3)
        file_rows = torch.cat([x[:, 0] for x, _, _, _ in batches]).tolist()
        assert file_rows == [i * WORLD_SIZE + rank for i in range(3 * B)]
        x, y, refs, state = batches[0]
        assert x.shape == (B, DOC_WIDTH - 1) and y.shape == (B, DOC_WIDTH - 1)
        assert refs.shape == (B, NUM_WINDOWS, WINDOW_LEN)
        assert torch.equal(refs[:, 0, 0], x[:, 0] + 1)  # refs come from the same row as the doc

def test_packed_loader_targets_are_inputs_shifted(packed_rows, monkeypatch):
    rows = packed_rows.astype(np.int64)
    rows[:, :DOC_WIDTH] = np.arange(DOC_WIDTH)  # distinct tokens along the doc, to check the shift
    np.save(nanochat.pack.packed_path("train"), rows.astype(np.uint16))
    x, y, _, _ = _load(monkeypatch, 0, B=2, n_batches=1)[0]
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert y[0, -1].item() == DOC_WIDTH - 1

def test_packed_loader_rows_per_step_independent_of_device_batch_size(packed_rows, monkeypatch):
    # one optimizer step of 4 rows per rank: 1 micro-batch of 4 or 2 micro-batches of 2 see the same rows
    big = _load(monkeypatch, 1, B=4, n_batches=1)
    small = _load(monkeypatch, 1, B=2, n_batches=2)
    assert big[0][0][:, 0].tolist() == torch.cat([x[:, 0] for x, _, _, _ in small]).tolist()

def test_packed_loader_resume_is_exact(packed_rows, monkeypatch):
    full = _load(monkeypatch, 2, B=3, n_batches=4)
    resumed = _load(monkeypatch, 2, B=3, n_batches=2, resume_state_dict=full[2][3])
    assert [x[:, 0].tolist() for x, _, _, _ in resumed] == [x[:, 0].tolist() for x, _, _, _ in full[2:]]

def test_packed_loader_wraps_to_next_epoch(packed_rows, monkeypatch):
    B = 3  # STREAM_LEN=10 rows per rank: 3 full batches, then the partial 4th is dropped and it wraps
    batches = _load(monkeypatch, 0, B, n_batches=4)
    assert [s["epoch"] for _, _, _, s in batches] == [1, 1, 1, 2]
    assert batches[3][0][:, 0].tolist() == batches[0][0][:, 0].tolist()

# -----------------------------------------------------------------------------
# embedding resume

class _FakeEmbedder:
    """ Stands in for the SentenceTransformer: embeds a text as a one-hot of its length, counting calls. """
    dim = 8
    def __init__(self):
        self.n_encoded = 0
    def get_sentence_embedding_dimension(self):
        return self.dim
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.n_encoded += len(texts)
        return np.eye(self.dim, dtype=np.float32)[[len(t) % self.dim for t in texts]]

class _FakeTokenizer:
    def decode(self, ids):
        return "x" * len(ids)

def test_embed_rows_resumes_after_interruption(tmp_path):
    n, batch_size = 40, 8
    out_path = str(tmp_path / "emb.npy")
    row_ids_fn = lambda i: [0] * (i % 5 + 1)
    complete = _FakeEmbedder()
    embed_rows_to_memmap(row_ids_fn, n, str(tmp_path / "ref.npy"), complete, _FakeTokenizer(), batch_size=batch_size)
    expected = np.load(tmp_path / "ref.npy")

    # simulate a run that crashed after writing its first 3 batches, then rerun: only the remaining rows get embedded
    partial = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, _FakeEmbedder.dim))
    partial[:3 * batch_size] = expected[:3 * batch_size]
    partial.flush()
    del partial
    rerun = _FakeEmbedder()
    embed_rows_to_memmap(row_ids_fn, n, out_path, rerun, _FakeTokenizer(), batch_size=batch_size)
    assert rerun.n_encoded == n - 3 * batch_size
    assert np.array_equal(np.load(out_path), expected)
