"""
Test the SFT/eval wiki-ref plumbing: distinct-article selection, per-task retrieval queries,
and attaching precomputed refs to tasks (all hermetic, no network / GPU / cached datasets).

python -m pytest tests/test_task_refs.py -v
"""

import json

import numpy as np
import pyarrow as pa
import pytest

import tasks.common as common
from tasks.common import Task, TaskMixture, HubDataset, attach_refs, queries_hash
from nanochat.wiki import top_distinct_articles


def test_top_distinct_articles():
    chunk_article = np.array([0, 0, 1, 1, 2, 3, 3, 4])
    raw_idx = np.array([[1, 0, 3, 2, 5, 4, 6, 7]])          # score-sorted; articles 0,0,1,1,3,2,3,4
    raw_scores = np.array([[.9, .8, .7, .6, .5, .4, .3, .2]], dtype=np.float32)
    ids, scores = top_distinct_articles(raw_idx, raw_scores, chunk_article, 4)
    assert ids.tolist() == [[1, 3, 5, 4]]                     # best chunk of articles 0, 1, 3, 2 in score order
    assert np.allclose(scores, [[.9, .7, .5, .4]])
    assert len(set(chunk_article[ids[0]].tolist())) == 4
    ids, scores = top_distinct_articles(raw_idx, raw_scores, chunk_article, 6)  # only 5 distinct articles exist
    assert ids[0, 5] == -1 and scores[0, 5] < -1e8


def _fake_ds(monkeypatch, module, **columns):
    monkeypatch.setattr(module, "load_hub_dataset", lambda *a, **k: HubDataset(pa.table(columns)))


def test_retrieval_queries_use_only_the_prompt(monkeypatch):
    import tasks.smoltalk, tasks.mmlu, tasks.arc, tasks.gsm8k, tasks.humaneval
    messages = [[{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER1"},
                 {"role": "assistant", "content": "ANSWER"}, {"role": "user", "content": "USER2"},
                 {"role": "assistant", "content": "ANSWER2"}]]
    _fake_ds(monkeypatch, tasks.smoltalk, messages=messages)
    assert tasks.smoltalk.SmolTalk("train").retrieval_query(0) == "USER1"

    _fake_ds(monkeypatch, tasks.mmlu, question=["Q?"], choices=[["c1", "c2", "c3", "c4"]], answer=[1], subject=["s"])
    assert tasks.mmlu.MMLU("all", "test").retrieval_query(0) == "Q?"

    _fake_ds(monkeypatch, tasks.arc, question=["Q?"], choices=[{"text": ["c1"], "label": ["A"]}], answerKey=["A"])
    assert tasks.arc.ARC("ARC-Easy", "test").retrieval_query(0) == "Q?"

    _fake_ds(monkeypatch, tasks.gsm8k, question=["Q?"], answer=["work #### 4"])
    assert tasks.gsm8k.GSM8K("main", "test").retrieval_query(0) == "Q?"

    _fake_ds(monkeypatch, tasks.humaneval, prompt=["def f():"], canonical_solution=["pass"], entry_point=["f"], test=["t"])
    assert tasks.humaneval.HumanEval().retrieval_query(0) == "def f():"


class ToyTask(Task):
    def __init__(self, n=10, key="toy", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.ref_key = key

    def num_examples(self):
        return self.n

    def get_example(self, index):
        return {"i": index}

    def retrieval_query(self, index):
        return f"{self.ref_key} query {index}"


def _write_refs(tmp_path, monkeypatch, key, n, n_ref=8, queries_sha256=None, n_chunks=10_000):
    monkeypatch.setattr(common, "REFS_DIR", str(tmp_path))
    monkeypatch.setattr(common, "wiki_num_chunks", lambda: 10_000)
    ids = (np.arange(n)[:, None] * 100 + np.arange(n_ref)[None, :]).astype(np.int32)  # ids[i, j] = 100 i + j
    np.save(tmp_path / f"{key}_ids.npy", ids)
    sha = queries_hash(ToyTask(n, key)) if queries_sha256 is None else queries_sha256
    (tmp_path / f"{key}.json").write_text(json.dumps({"n": n, "n_ref": n_ref, "n_chunks": n_chunks, "queries_sha256": sha}))


def test_attach_refs_without_attach_has_no_refs():
    assert "ref_ids" not in ToyTask()[3]


def test_attach_refs_follows_slicing(tmp_path, monkeypatch):
    _write_refs(tmp_path, monkeypatch, "toy", 10)
    task = attach_refs(ToyTask(n=10, start=2, stop=9, step=3))  # physical indices 2, 5, 8
    assert [task[i]["i"] for i in range(len(task))] == [2, 5, 8]
    assert [task[i]["ref_ids"][0] for i in range(len(task))] == [200, 500, 800]
    assert len(task[0]["ref_ids"]) == 8


def test_attach_refs_through_mixture(tmp_path, monkeypatch):
    _write_refs(tmp_path, monkeypatch, "a", 4)
    _write_refs(tmp_path, monkeypatch, "b", 3)
    mix = attach_refs(TaskMixture([ToyTask(4, "a"), ToyTask(3, "b"), ToyTask(4, "a")]))
    assert len(mix) == 11
    for i in range(len(mix)):
        conv = mix[i]
        assert conv["ref_ids"][0] == conv["i"] * 100


def test_attach_refs_rejects_stale_refs(tmp_path, monkeypatch):
    _write_refs(tmp_path, monkeypatch, "toy", 9)  # wrong length
    with pytest.raises(AssertionError, match="rerun"):
        attach_refs(ToyTask(n=10))
    _write_refs(tmp_path, monkeypatch, "toy", 10, queries_sha256="0" * 64)  # dataset order changed
    with pytest.raises(AssertionError, match="rerun"):
        attach_refs(ToyTask(n=10))
    _write_refs(tmp_path, monkeypatch, "toy", 10, n_chunks=9_999)  # wiki rebuilt since
    with pytest.raises(AssertionError, match="rerun"):
        attach_refs(ToyTask(n=10))


def test_search_merge_matches_old_inline_loop(tmp_path, monkeypatch):
    import scripts.wiki_retrieve as wr
    rng = np.random.default_rng(0)
    R, world_size, n_chunks = 3, 2, 600
    n = R * wr.NUM_WINDOWS
    chunk_article = rng.integers(0, 30, n_chunks)  # fewer articles than ARTICLE_K: duplicates everywhere, padding too
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    np.save(wiki_dir / "meta.npy", np.stack([chunk_article, np.zeros(n_chunks, dtype=np.int64)], axis=1))
    monkeypatch.setattr(wr, "WIKI_DIR", str(wiki_dir))
    monkeypatch.setattr(wr, "PACK_DIR", str(tmp_path))
    monkeypatch.setattr(wr, "packed_path", lambda split: str(tmp_path / f"{split}.npy"))
    np.save(tmp_path / "val.npy", np.zeros((R, 1), dtype=np.uint16))
    shards = []
    for r in range(world_size):  # each shard: its own chunk range, score-sorted local top-CAND_K
        lo = r * n_chunks // world_size
        idx = np.stack([rng.choice(n_chunks // world_size, wr.CAND_K, replace=False) + lo for _ in range(n)]).astype(np.int64)
        scores = -np.sort(-rng.random((n, wr.CAND_K)).astype(np.float32), axis=1)
        idx_path, scores_path = wr._shard_paths("val", r)
        np.save(idx_path, idx)
        np.save(scores_path, scores)
        shards.append((idx, scores))

    # the pre-refactor merge + collapse, inline
    idx_cat = np.concatenate([s[0] for s in shards], axis=1)
    score_cat = np.concatenate([s[1] for s in shards], axis=1)
    order = np.argsort(-score_cat, axis=1)[:, :wr.CAND_K]
    raw_idx = np.take_along_axis(idx_cat, order, axis=1)
    raw_scores = np.take_along_axis(score_cat, order, axis=1)
    raw_articles = chunk_article[raw_idx]
    want_ids = np.full((n, wr.ARTICLE_K), -1, dtype=np.int32)
    want_scores = np.full((n, wr.ARTICLE_K), -1e9, dtype=np.float32)
    for i in range(n):
        _, first = np.unique(raw_articles[i], return_index=True)
        keep = np.sort(first)[:wr.ARTICLE_K]
        want_ids[i, :len(keep)] = raw_idx[i, keep]
        want_scores[i, :len(keep)] = raw_scores[i, keep]

    wr.search_merge("val", world_size=world_size, merge_batch=7)  # odd batch size: exercises the batch boundaries
    got_ids = np.load(tmp_path / "val_cand_ids.npy").reshape(n, wr.ARTICLE_K)
    got_scores = np.load(tmp_path / "val_cand_scores.npy").reshape(n, wr.ARTICLE_K)
    assert (want_ids == -1).any()  # the padding path is covered
    assert np.array_equal(got_ids, want_ids)
    assert np.array_equal(got_scores, want_scores)
