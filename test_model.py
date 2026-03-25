"""Unit tests for the LLM model."""

import pytest
import torch

from model import LLM, MultiHeadAttention, FeedForward, TransformerBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 32
BATCH_SIZE = 2
SEQ_LEN = 16


@pytest.fixture
def small_model():
    return LLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    )


# ---------------------------------------------------------------------------
# MultiHeadAttention tests
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out = attn(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_output_shape_with_mask(self):
        attn = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).unsqueeze(0).unsqueeze(0)
        out = attn(x, mask=mask)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_invalid_heads_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=64, num_heads=3)  # 64 not divisible by 3


# ---------------------------------------------------------------------------
# FeedForward tests
# ---------------------------------------------------------------------------

class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(d_model=D_MODEL, d_ff=D_MODEL * 4)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out = ff(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# TransformerBlock tests
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_MODEL * 4)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        out = block(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# LLM tests
# ---------------------------------------------------------------------------

class TestLLM:
    def test_forward_output_shape(self, small_model):
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits = small_model(input_ids)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    def test_num_parameters_positive(self, small_model):
        assert small_model.num_parameters() > 0

    def test_weight_tying(self, small_model):
        """Token embedding and lm_head should share weights."""
        assert small_model.token_embedding.weight is small_model.lm_head.weight

    def test_seq_len_exceeds_max_raises(self, small_model):
        input_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN + 1))
        with pytest.raises(AssertionError):
            small_model(input_ids)

    def test_generate_output_length(self, small_model):
        prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
        max_new = 10
        output = small_model.generate(prompt, max_new_tokens=max_new)
        assert output.shape == (1, 4 + max_new)

    def test_generate_temperature(self, small_model):
        """Greedy (temperature → 0) should give a fixed sequence."""
        torch.manual_seed(42)
        prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
        # Very low temperature → near-deterministic
        torch.manual_seed(0)
        out1 = small_model.generate(prompt, max_new_tokens=5, temperature=1e-9)
        torch.manual_seed(0)
        out2 = small_model.generate(prompt, max_new_tokens=5, temperature=1e-9)
        assert torch.equal(out1, out2)

    def test_generate_top_k(self, small_model):
        prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
        output = small_model.generate(prompt, max_new_tokens=5, top_k=5)
        assert output.shape == (1, 4 + 5)

    def test_causal_mask(self, small_model):
        """Future tokens must not affect past positions."""
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        small_model.eval()
        with torch.no_grad():
            logits_full = small_model(input_ids)
            logits_prefix = small_model(input_ids[:, :8])
        # The logits for the first 7 positions should match
        assert torch.allclose(logits_full[:, :7, :], logits_prefix[:, :7, :], atol=1e-5)
