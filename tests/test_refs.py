"""
Tests for the "cheat sheet" reference modes in nanochat.gpt.GPT: ref_mode="prefix"
(baseline: refs concatenated before idx) and ref_mode="encoder" (cheat_sheet: refs
go through a separate encoder, attended to via cross-attention).

Hermetic, CPU-only, forces the SDPA attention fallback since FA3 requires CUDA.

python -m pytest tests/test_refs.py -v
"""
import torch
import pytest

import nanochat.flash_attention as fa_module
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache
from nanochat.common import COMPUTE_DTYPE


def setup_module(module):
    fa_module._override_impl = 'sdpa'
    fa_module.USE_FA3 = False


def teardown_module(module):
    fa_module._override_impl = None
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


SEQ_LEN = 32
N_REF, REF_LEN = 2, 16
BASE_KW = dict(vocab_size=256, n_layer=4, n_head=2, n_kv_head=2, n_embd=32, window_pattern='L')


def make_model(ref_mode, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = GPTConfig(sequence_len=SEQ_LEN, ref_mode=ref_mode, n_ref=N_REF, ref_len=REF_LEN, **BASE_KW, **kw)
    with torch.device('meta'):
        model = GPT(cfg)
    model.to_empty(device='cpu')
    model.init_weights()
    model.eval()
    return model


def random_batch(batch_size=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 256, (batch_size, SEQ_LEN), generator=g)
    tgt = torch.randint(0, 256, (batch_size, SEQ_LEN), generator=g)
    refs = torch.randint(0, 256, (batch_size, N_REF, REF_LEN), generator=g)
    return idx, tgt, refs


class TestOldCheckpointsStillLoad:
    def test_none_mode_state_dict_round_trips(self):
        """A ref_mode='none' model's state dict has no new keys and loads strict=True
        into a freshly constructed model, i.e. old checkpoints keep working."""
        m1 = make_model('none', seed=1)
        sd = m1.state_dict()
        m2 = make_model('none', seed=2)
        m2.load_state_dict(sd, strict=True)


class TestBookkeeping:
    @pytest.mark.parametrize('ref_mode,kw', [
        ('none', {}),
        ('prefix', {}),
        ('encoder', dict(n_enc_layer=2, cross_every=1)),
        ('encoder', dict(n_enc_layer=2, cross_every=2)),
    ])
    def test_setup_optimizer_and_scaling_params(self, ref_mode, kw):
        """Every parameter must land in exactly one optimizer group, and num_scaling_params's
        internal total must match the real parameter count (both methods self-assert this)."""
        m = make_model(ref_mode, **kw)
        m.setup_optimizer()
        m.num_scaling_params()

    def test_estimate_flops_positive(self):
        for ref_mode, kw in [('none', {}), ('prefix', {}), ('encoder', dict(n_enc_layer=2, cross_every=1))]:
            m = make_model(ref_mode, **kw)
            assert m.estimate_flops() > 0

    def test_estimate_flops_counts_encoder(self):
        """Adding encoder layers must add their matmul FLOPs (6/param/token, amortized over the
        t doc tokens by P/t) plus their non-causal self-attention FLOPs."""
        m0 = make_model('encoder', n_enc_layer=0, cross_every=1)
        m2 = make_model('encoder', n_enc_layer=2, cross_every=1)
        P, t = N_REF * REF_LEN, SEQ_LEN
        h, q = m2.config.n_head, m2.config.n_embd // m2.config.n_head
        encoder_params = sum(p.numel() for p in m2.encoder.parameters())
        expected = 6 * encoder_params * (P / t) + 12 * h * q * REF_LEN * 2 * (P / t)
        assert m2.estimate_flops() - m0.estimate_flops() == pytest.approx(expected)

    def test_decode_flops_exclude_encoder(self):
        """Refs are encoded once at prefill, so decode cost must not depend on encoder depth,
        while prefill cost must."""
        m0 = make_model('encoder', n_enc_layer=0, cross_every=1)
        m2 = make_model('encoder', n_enc_layer=2, cross_every=1)
        assert m0.estimate_decode_flops(16) == m2.estimate_decode_flops(16)
        assert m2.estimate_prefill_flops(16) > m0.estimate_prefill_flops(16)

    def test_prefix_short_window_matches_plain_decoder(self):
        """In prefix mode only the long window grows to cover the refs; S layers keep the
        plain decoder's window."""
        kw = dict(window_pattern='SL')
        cfg = lambda ref_mode: GPTConfig(sequence_len=1024, ref_mode=ref_mode, n_ref=N_REF, ref_len=REF_LEN, **{**BASE_KW, **kw})
        m_none, m_prefix = GPT(cfg('none')), GPT(cfg('prefix'))
        assert m_prefix.window_sizes[0] == m_none.window_sizes[0]
        assert m_prefix.window_sizes[-1] == (1024 + N_REF * REF_LEN, 0)


class TestEncoderIgnoresRefsAtInit:
    def test_cross_attn_zero_init_means_refs_dont_matter_yet(self):
        """cross_attn.c_proj is zero-initialized, so at init the decoder's output must be
        identical regardless of what references it's given."""
        m = make_model('encoder', n_enc_layer=2, cross_every=1)
        idx, _, _ = random_batch()
        refs_a = torch.randint(0, 256, (2, N_REF, REF_LEN))
        refs_b = torch.randint(0, 256, (2, N_REF, REF_LEN))
        logits_a = m(idx, refs=refs_a)
        logits_b = m(idx, refs=refs_b)
        assert torch.allclose(logits_a, logits_b)

    def test_after_training_step_refs_do_matter(self):
        """Once cross_attn.c_proj moves off zero, different references must give different logits."""
        m = make_model('encoder', n_enc_layer=2, cross_every=1)
        idx, tgt, refs = random_batch()
        loss = m(idx, tgt, refs=refs)
        loss.backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1.0)
        refs_b = torch.randint(0, 256, refs.shape)
        logits_a = m(idx, refs=refs)
        logits_b = m(idx, refs=refs_b)
        assert not torch.allclose(logits_a, logits_b)


class TestPrefixEqualsConcatenation:
    def test_prefix_matches_plain_concat_sliced(self):
        """ref_mode='prefix' must be exactly equivalent to a ref_mode='none' model run on
        cat(refs, idx), sliced back down to the doc positions, given the same weights."""
        idx, _, refs = random_batch()
        torch.manual_seed(7)
        m_prefix = make_model('prefix', seed=7)
        cfg_none = GPTConfig(sequence_len=SEQ_LEN + N_REF * REF_LEN, ref_mode='none', **BASE_KW)
        with torch.device('meta'):
            m_none = GPT(cfg_none)
        m_none.to_empty(device='cpu')
        torch.manual_seed(7)
        m_none.init_weights()
        m_none.eval()

        logits_prefix = m_prefix(idx, refs=refs)
        cat = torch.cat([refs.view(refs.size(0), -1), idx], dim=1)
        logits_cat = m_none(cat)
        logits_cat_sliced = logits_cat[:, N_REF * REF_LEN:]
        assert torch.allclose(logits_prefix, logits_cat_sliced, atol=1e-4)


class TestKVCacheMatchesFullForward:
    @pytest.mark.parametrize('ref_mode,kw', [
        ('prefix', {}),
        ('encoder', dict(n_enc_layer=2, cross_every=1)),
    ])
    def test_prefill_then_decode_matches_full_pass(self, ref_mode, kw):
        m = make_model(ref_mode, **kw)
        idx, _, refs = random_batch(batch_size=1)
        full_logits = m(idx, refs=refs)

        cache_len = (N_REF * REF_LEN + SEQ_LEN) if ref_mode == 'prefix' else SEQ_LEN
        kv = KVCache(batch_size=1, num_heads=m.config.n_kv_head, seq_len=cache_len,
                     head_dim=m.config.n_embd // m.config.n_head, num_layers=m.config.n_layer,
                     device='cpu', dtype=COMPUTE_DTYPE)
        prompt, last = idx[:, :-1], idx[:, -1:]
        m(prompt, kv_cache=kv, refs=refs)
        decode_logits = m(last, kv_cache=kv)
        assert torch.allclose(full_logits[:, -1, :], decode_logits[:, -1, :], atol=1e-2)

    def test_encoder_mode_caches_cross_kv_once(self):
        """Cross-attn k/v must be cached at prefill (one entry per cross-attn layer) and
        not recomputed on decode steps."""
        m = make_model('encoder', n_enc_layer=2, cross_every=1)
        idx, _, refs = random_batch(batch_size=1)
        kv = KVCache(batch_size=1, num_heads=m.config.n_kv_head, seq_len=SEQ_LEN,
                     head_dim=m.config.n_embd // m.config.n_head, num_layers=m.config.n_layer,
                     device='cpu', dtype=COMPUTE_DTYPE)
        prompt, last = idx[:, :-1], idx[:, -1:]
        m(prompt, kv_cache=kv, refs=refs)
        assert sorted(kv.cross_kv.keys()) == list(range(m.config.n_layer))
        cached_before = {k: (v[0].clone(), v[1].clone()) for k, v in kv.cross_kv.items()}
        m(last, kv_cache=kv)  # refs=None: must reuse cached cross-attn k/v
        for layer_idx, (k_before, v_before) in cached_before.items():
            k_after, v_after = kv.cross_kv[layer_idx]
            assert torch.equal(k_before, k_after) and torch.equal(v_before, v_after)


class TestRefsRequired:
    def test_missing_refs_raises_for_non_none_modes(self):
        idx, _, _ = random_batch()
        for ref_mode, kw in [('prefix', {}), ('encoder', dict(n_enc_layer=2, cross_every=1))]:
            m = make_model(ref_mode, **kw)
            with pytest.raises(AssertionError):
                m(idx)


class ByteTokenizer:
    """Just enough tokenizer for Engine: special tokens sit above the model's vocab, so
    greedy decoding never samples them (no tool-use state machine involvement)."""
    def encode_special(self, s):
        return {"<|python_start|>": 256, "<|python_end|>": 257, "<|output_start|>": 258,
                "<|output_end|>": 259, "<|assistant_end|>": 260}[s]

    def get_bos_token_id(self):
        return 261


class TestEngineWithRefs:
    @pytest.mark.parametrize('ref_mode,kw', [
        ('prefix', {}),
        ('encoder', dict(n_enc_layer=2, cross_every=1)),
    ])
    def test_engine_multi_sample_matches_naive_greedy(self, ref_mode, kw):
        """Engine does a batch=1 prefill and then clones the KV cache to num_samples rows, so
        this exercises KVCache.prefill copying cross-attn k/v (encoder) and the cache being
        sized for the reference prefix (prefix)."""
        from nanochat.engine import Engine
        m = make_model(ref_mode, **kw)
        # Move the zero-init output projections off zero so refs (and attention) actually matter
        with torch.no_grad():
            for name, p in m.named_parameters():
                if 'c_proj' in name:
                    torch.nn.init.uniform_(p, -0.05, 0.05)
        _, _, refs = random_batch(batch_size=1)
        prompt = [7, 42, 99, 3]
        max_tokens = 6

        ids = torch.tensor([prompt])
        expected = []
        for _ in range(max_tokens):
            next_id = m(ids, refs=refs)[:, -1, :].argmax(dim=-1, keepdim=True)
            expected.append(next_id.item())
            ids = torch.cat([ids, next_id], dim=1)

        engine = Engine(m, ByteTokenizer())
        rows, _ = engine.generate_batch(prompt, num_samples=2, max_tokens=max_tokens, temperature=0.0, refs=refs[0].tolist())
        for row in rows:
            assert row[len(prompt):] == expected
