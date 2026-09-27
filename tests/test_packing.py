"""
Tests for offline row packing (nanochat.dataloader.BestFitPacker, shared by the live
training loader and nanochat.pack), wiki chunking (nanochat.wiki.split_list_with_overlap),
and retrieval dedup (scripts.wiki_retrieve.dedup_row).

Hermetic: trains a tiny throwaway tokenizer in-process, no dependency on ~/.cache/nanochat.

python -m pytest tests/test_packing.py -v
"""
import numpy as np
import pytest

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
from nanochat.dataloader import BestFitPacker
from nanochat.wiki import split_list_with_overlap
from scripts.wiki_retrieve import dedup_row, ARTICLE_K

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
    chunk_article = {10: 1, 11: 1, 12: 1, 20: 2, 21: 2, 22: 2, 30: 3, 31: 3, 32: 3}
    chunk, score = dedup_row(cand_ids, cand_scores, chunk_article)
    assert chunk == [10, 20, 30]

def test_dedup_row_resolves_conflict_by_score_and_terminates():
    # windows 0 and 1 both want article 1 at rank 0; window 1 has the higher score and should win
    cand_ids = np.array([[10, 13], [11, 14]])
    cand_scores = np.array([[0.5, 0.1], [0.9, 0.2]])
    chunk_article = {10: 1, 13: 5, 11: 1, 14: 6}
    chunk, score = dedup_row(cand_ids, cand_scores, chunk_article)
    assert chunk[1] == 11  # window 1 keeps its top choice (higher score)
    assert chunk[0] == 13  # window 0 was displaced to its next candidate
    assert chunk_article[chunk[0]] != chunk_article[chunk[1]]

def test_dedup_row_no_duplicate_articles_with_heavy_overlap():
    num_windows = 8
    rng = np.random.default_rng(0)
    cand_ids = np.arange(num_windows * ARTICLE_K).reshape(num_windows, ARTICLE_K)
    cand_scores = -np.sort(-rng.random((num_windows, ARTICLE_K)), axis=1)  # descending, like real candidates
    # every window's 32 candidates cycle through all 16 possible articles (32 % 16 == 0),
    # so conflicts are frequent but always resolvable
    chunk_article = {int(c): int(c) % 16 for c in cand_ids.flatten()}
    chunk, score = dedup_row(cand_ids, cand_scores, chunk_article)
    assert all(c >= 0 for c in chunk)  # never runs out of candidates in this regime
    assigned_articles = [chunk_article[c] for c in chunk]
    assert len(assigned_articles) == len(set(assigned_articles))
