# NOTE: This file is specific to PyRin-c/CTranslate2 (feature/add_support_parakeet-ctc).
# Not present in upstream OpenNMT/CTranslate2.
#
# Tests for:
#   - conformer_spec.py  (Spec structure / validation)
#   - ctranslate2.models.Parakeet  (runtime E2E — skipped when no model available)
#   - NemoParakeetConverter (conversion — skipped when NeMo is unavailable)

import os

import numpy as np
import pytest

import ctranslate2
from ctranslate2.specs import common_spec
from ctranslate2.specs.conformer_spec import (
    Conv2dSpec,
    LSTMLayerSpec,
    ConvModuleSpec,
    RelPosAttentionSpec,
    ConformerBlockSpec,
    ConformerPreEncoderSpec,
    ConformerEncoderSpec,
    CTCDecoderSpec,
    TDTDecoderSpec,
    TDTJointSpec,
    ParakeetModelSpec,
)
from ctranslate2.specs.model_spec import OPTIONAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear(rows, cols):
    spec = common_spec.LinearSpec()
    spec.weight = np.zeros((rows, cols), dtype=np.float32)
    return spec


def _make_layer_norm(size):
    spec = common_spec.LayerNormSpec()
    spec.gamma = np.ones(size, dtype=np.float32)
    spec.beta = np.zeros(size, dtype=np.float32)
    return spec


def _make_conv1d(out_ch, in_ch, k):
    spec = common_spec.Conv1DSpec()
    spec.weight = np.zeros((out_ch, in_ch, k), dtype=np.float32)
    return spec


def _fill_conv_module(spec, d_model=8, depthwise_k=3):
    spec.norm = _make_layer_norm(d_model)
    spec.pointwise_conv1 = _make_linear(2 * d_model, d_model)
    spec.depthwise_conv = _make_conv1d(d_model, 1, depthwise_k)
    spec.pointwise_conv2 = _make_linear(d_model, d_model)


def _fill_rel_pos_attention(spec, d_model=8):
    spec.linear_q = _make_linear(d_model, d_model)
    spec.linear_k = _make_linear(d_model, d_model)
    spec.linear_v = _make_linear(d_model, d_model)
    spec.linear_out = _make_linear(d_model, d_model)
    spec.linear_pos = _make_linear(d_model, d_model)
    num_heads, head_dim = 1, d_model
    spec.pos_bias_u = np.zeros((num_heads, head_dim), dtype=np.float32)
    spec.pos_bias_v = np.zeros((num_heads, head_dim), dtype=np.float32)


def _fill_conformer_block(spec, d_model=8, ff_dim=16):
    spec.norm_ff1 = _make_layer_norm(d_model)
    spec.ff1_linear1 = _make_linear(ff_dim, d_model)
    spec.ff1_linear2 = _make_linear(d_model, ff_dim)
    spec.norm_mha = _make_layer_norm(d_model)
    _fill_rel_pos_attention(spec.mha, d_model)
    spec.norm_conv = _make_layer_norm(d_model)
    _fill_conv_module(spec.conv, d_model)
    spec.norm_ff2 = _make_layer_norm(d_model)
    spec.ff2_linear1 = _make_linear(ff_dim, d_model)
    spec.ff2_linear2 = _make_linear(d_model, ff_dim)
    spec.norm_out = _make_layer_norm(d_model)


def _fill_pre_encoder(spec, d_model=8, sub_ch=4):
    # Shapes mirror the real model but with tiny dimensions
    spec.conv0.weight = np.zeros((sub_ch, 1, 3, 3), dtype=np.float32)
    spec.conv2.weight = np.zeros((sub_ch, 1, 3, 3), dtype=np.float32)
    spec.conv3.weight = np.zeros((sub_ch, sub_ch, 1, 1), dtype=np.float32)
    spec.conv5.weight = np.zeros((sub_ch, 1, 3, 3), dtype=np.float32)
    spec.conv6.weight = np.zeros((sub_ch, sub_ch, 1, 1), dtype=np.float32)
    # out Linear: input size = sub_ch * 10 (arbitrary, just needs a weight)
    spec.out = _make_linear(d_model, sub_ch * 10)


def _build_minimal_parakeet_spec(num_layers=1, d_model=8, vocab_size=5, num_durations=2):
    """Build a fully-populated ParakeetModelSpec with tiny dimensions."""
    spec = ParakeetModelSpec.__new__(ParakeetModelSpec)
    # Bypass __init__ to override num_layers=24 default
    from ctranslate2.specs.model_spec import ModelSpec
    ModelSpec.__init__(spec)
    spec.vocab_size = None
    spec.num_durations = None

    spec.encoder = ConformerEncoderSpec(num_layers=num_layers)
    spec.ctc_decoder = OPTIONAL
    spec.tdt_decoder = TDTDecoderSpec()

    # Fill scalars
    spec.vocab_size = np.int32(vocab_size)
    spec.num_durations = np.int32(num_durations)

    # Fill encoder
    _fill_pre_encoder(spec.encoder.pre_encode, d_model=d_model)
    for layer in spec.encoder.layer:
        _fill_conformer_block(layer, d_model=d_model)

    # Fill TDT decoder
    pred_dim = d_model
    hidden = d_model
    joint_dim = d_model

    spec.tdt_decoder.embedding.weight = np.zeros((vocab_size, pred_dim), dtype=np.float32)

    spec.tdt_decoder.lstm_layer_0.weight_ih = np.zeros((4 * hidden, pred_dim), dtype=np.float32)
    spec.tdt_decoder.lstm_layer_0.weight_hh = np.zeros((4 * hidden, hidden), dtype=np.float32)
    spec.tdt_decoder.lstm_layer_0.bias = np.zeros(4 * hidden, dtype=np.float32)

    spec.tdt_decoder.lstm_layer_1.weight_ih = np.zeros((4 * hidden, hidden), dtype=np.float32)
    spec.tdt_decoder.lstm_layer_1.weight_hh = np.zeros((4 * hidden, hidden), dtype=np.float32)
    spec.tdt_decoder.lstm_layer_1.bias = np.zeros(4 * hidden, dtype=np.float32)

    spec.tdt_decoder.joint.enc_proj = _make_linear(joint_dim, d_model)
    spec.tdt_decoder.joint.pred_proj = _make_linear(joint_dim, hidden)
    spec.tdt_decoder.joint.output_head = _make_linear(vocab_size + num_durations, joint_dim)

    return spec


# ---------------------------------------------------------------------------
# Conv2dSpec
# ---------------------------------------------------------------------------

class TestConv2dSpec:
    def test_validate_weight_only(self):
        spec = Conv2dSpec()
        spec.weight = np.zeros((4, 1, 3, 3), dtype=np.float32)
        spec.validate()  # must not raise

    def test_validate_with_bias(self):
        spec = Conv2dSpec()
        spec.weight = np.zeros((4, 1, 3, 3), dtype=np.float32)
        spec.bias = np.zeros(4, dtype=np.float32)
        spec.validate()

    def test_validate_missing_weight_raises(self):
        spec = Conv2dSpec()
        with pytest.raises(ValueError):
            spec.validate()


# ---------------------------------------------------------------------------
# LSTMLayerSpec
# ---------------------------------------------------------------------------

class TestLSTMLayerSpec:
    def test_validate_with_all_weights(self):
        spec = LSTMLayerSpec()
        spec.weight_ih = np.zeros((16, 4), dtype=np.float32)
        spec.weight_hh = np.zeros((16, 4), dtype=np.float32)
        spec.bias = np.zeros(16, dtype=np.float32)
        spec.validate()

    def test_validate_without_bias(self):
        spec = LSTMLayerSpec()
        spec.weight_ih = np.zeros((16, 4), dtype=np.float32)
        spec.weight_hh = np.zeros((16, 4), dtype=np.float32)
        # bias is OPTIONAL — leaving it unset is fine
        spec.validate()

    def test_validate_missing_weight_ih_raises(self):
        spec = LSTMLayerSpec()
        spec.weight_hh = np.zeros((16, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            spec.validate()


# ---------------------------------------------------------------------------
# ConvModuleSpec
# ---------------------------------------------------------------------------

class TestConvModuleSpec:
    def test_validate(self):
        spec = ConvModuleSpec()
        _fill_conv_module(spec, d_model=8)
        spec.validate()


# ---------------------------------------------------------------------------
# RelPosAttentionSpec
# ---------------------------------------------------------------------------

class TestRelPosAttentionSpec:
    def test_validate(self):
        spec = RelPosAttentionSpec()
        _fill_rel_pos_attention(spec, d_model=8)
        spec.validate()


# ---------------------------------------------------------------------------
# ConformerBlockSpec
# ---------------------------------------------------------------------------

class TestConformerBlockSpec:
    def test_validate(self):
        spec = ConformerBlockSpec()
        _fill_conformer_block(spec, d_model=8)
        spec.validate()


# ---------------------------------------------------------------------------
# ParakeetModelSpec — metadata
# ---------------------------------------------------------------------------

class TestParakeetModelSpec:
    def test_name(self):
        spec = ParakeetModelSpec()
        assert spec.name == "ParakeetModel"

    def test_revision(self):
        spec = ParakeetModelSpec()
        assert spec.revision == 1

    def test_ctc_decoder_is_optional(self):
        spec = ParakeetModelSpec()
        assert spec.ctc_decoder == OPTIONAL

    def test_encoder_default_num_layers(self):
        spec = ParakeetModelSpec()
        assert len(spec.encoder.layer) == 24

    def test_validate_minimal_spec(self):
        spec = _build_minimal_parakeet_spec(num_layers=1, d_model=8)
        spec.validate()  # must not raise

    def test_validate_missing_vocab_size_raises(self):
        spec = _build_minimal_parakeet_spec(num_layers=1, d_model=8)
        spec.vocab_size = None
        with pytest.raises(ValueError):
            spec.validate()

    def test_validate_missing_num_durations_raises(self):
        spec = _build_minimal_parakeet_spec(num_layers=1, d_model=8)
        spec.num_durations = None
        with pytest.raises(ValueError):
            spec.validate()


# ---------------------------------------------------------------------------
# CTC decode logic — pure-Python reference test
#
# The C++ ctc_decode_sequence() is a private method; we test the algorithm
# logic here to document and verify the expected behaviour (blank collapsing,
# repeat collapsing) independently of the runtime.
# ---------------------------------------------------------------------------

def _ctc_greedy_decode(log_probs_rows, blank_id):
    """Python reference: CTC greedy decode identical to C++ ctc_decode_sequence."""
    ids = []
    prev = blank_id
    for row in log_probs_rows:
        best = int(np.argmax(row))
        if best != blank_id and best != prev:
            ids.append(best)
        prev = best
    return ids


class TestCTCDecodeLogic:
    BLANK = 4  # vocab_size - 1

    def test_simple_tokens(self):
        # Each frame emits a distinct non-blank token → no collapsing.
        lp = np.full((3, 5), -10.0, dtype=np.float32)
        lp[0, 0] = 0.0  # token 0
        lp[1, 1] = 0.0  # token 1
        lp[2, 2] = 0.0  # token 2
        assert _ctc_greedy_decode(lp, self.BLANK) == [0, 1, 2]

    def test_blank_frames_removed(self):
        # Blank frames must be dropped entirely.
        lp = np.full((5, 5), -10.0, dtype=np.float32)
        lp[0, self.BLANK] = 0.0  # blank
        lp[1, 0] = 0.0           # token 0
        lp[2, self.BLANK] = 0.0  # blank
        lp[3, 1] = 0.0           # token 1
        lp[4, self.BLANK] = 0.0  # blank
        assert _ctc_greedy_decode(lp, self.BLANK) == [0, 1]

    def test_repeated_tokens_collapsed(self):
        # Consecutive frames with the same token → collapsed to one emission.
        lp = np.full((4, 5), -10.0, dtype=np.float32)
        lp[0, 0] = 0.0  # token 0
        lp[1, 0] = 0.0  # token 0 repeated → collapsed
        lp[2, 0] = 0.0  # token 0 repeated → collapsed
        lp[3, 1] = 0.0  # token 1
        assert _ctc_greedy_decode(lp, self.BLANK) == [0, 1]

    def test_repeated_token_separated_by_blank(self):
        # Same token, separated by a blank → two emissions.
        lp = np.full((3, 5), -10.0, dtype=np.float32)
        lp[0, 0] = 0.0           # token 0
        lp[1, self.BLANK] = 0.0  # blank
        lp[2, 0] = 0.0           # token 0 again
        assert _ctc_greedy_decode(lp, self.BLANK) == [0, 0]

    def test_all_blank(self):
        lp = np.full((5, 5), -10.0, dtype=np.float32)
        for t in range(5):
            lp[t, self.BLANK] = 0.0
        assert _ctc_greedy_decode(lp, self.BLANK) == []

    def test_empty_sequence(self):
        lp = np.zeros((0, 5), dtype=np.float32)
        assert _ctc_greedy_decode(lp, self.BLANK) == []


# ---------------------------------------------------------------------------
# Runtime E2E tests — skipped when no converted model is available
# ---------------------------------------------------------------------------

def _parakeet_model_dir():
    """Return the path set by the environment variable PARAKEET_CT2_MODEL_DIR,
    or skip the test if the variable is not set / the directory does not exist."""
    path = os.environ.get("PARAKEET_CT2_MODEL_DIR", "")
    if not path or not os.path.isdir(path):
        pytest.skip("PARAKEET_CT2_MODEL_DIR is not set or does not exist")
    return path


class TestParakeetRuntime:
    def test_model_loads_on_cpu(self):
        model_dir = _parakeet_model_dir()
        model = ctranslate2.models.Parakeet(model_dir, device="cpu")
        assert model.device == "cpu"
        assert model.model_is_loaded

    def test_encode_output_shape(self):
        model_dir = _parakeet_model_dir()
        model = ctranslate2.models.Parakeet(model_dir, device="cpu")
        batch, n_mels, time = 1, 80, 200
        mel = ctranslate2.StorageView.from_array(
            np.zeros((batch, n_mels, time), dtype=np.float32)
        )
        lengths = ctranslate2.StorageView.from_array(
            np.array([time], dtype=np.int32)
        )
        enc = model.encode(mel, lengths, to_cpu=True)
        arr = np.array(enc)
        assert arr.ndim == 3
        assert arr.shape[0] == batch
        assert arr.shape[2] == 1024  # Parakeet d_model

    def test_transcribe_ctc_returns_ids(self):
        model_dir = _parakeet_model_dir()
        model = ctranslate2.models.Parakeet(model_dir, device="cpu")
        mel = ctranslate2.StorageView.from_array(
            np.zeros((1, 80, 400), dtype=np.float32)
        )
        lengths = ctranslate2.StorageView.from_array(
            np.array([400], dtype=np.int32)
        )
        result = model.transcribe(mel, lengths, use_tdt=False)
        assert hasattr(result, "ids")
        assert isinstance(result.ids, list)
        assert len(result.ids) == 1  # batch size 1

    def test_transcribe_tdt_returns_start_frames(self):
        model_dir = _parakeet_model_dir()
        model = ctranslate2.models.Parakeet(model_dir, device="cpu")
        mel = ctranslate2.StorageView.from_array(
            np.zeros((1, 80, 400), dtype=np.float32)
        )
        lengths = ctranslate2.StorageView.from_array(
            np.array([400], dtype=np.int32)
        )
        result = model.transcribe(mel, lengths, use_tdt=True)
        assert hasattr(result, "ids")
        assert hasattr(result, "token_start_frames")
        # ids and token_start_frames must have the same number of tokens
        assert len(result.ids[0]) == len(result.token_start_frames[0])

    @pytest.mark.skipif(
        ctranslate2.get_cuda_device_count() == 0, reason="CUDA device required"
    )
    def test_transcribe_ctc_on_cuda(self):
        model_dir = _parakeet_model_dir()
        model = ctranslate2.models.Parakeet(model_dir, device="cuda")
        mel = ctranslate2.StorageView.from_array(
            np.zeros((1, 80, 400), dtype=np.float32)
        )
        lengths = ctranslate2.StorageView.from_array(
            np.array([400], dtype=np.int32)
        )
        result = model.transcribe(mel, lengths, use_tdt=False)
        assert isinstance(result.ids, list)


# ---------------------------------------------------------------------------
# Converter tests — skipped when NeMo is unavailable
# ---------------------------------------------------------------------------

class TestNemoParakeetConverter:
    @pytest.fixture(autouse=True)
    def _skip_without_nemo(self):
        try:
            import nemo  # noqa: F401
        except ImportError:
            pytest.skip("NeMo is not installed")

    def test_converter_import(self):
        from ctranslate2.converters.nemo_parakeet import NemoParakeetConverter
        assert NemoParakeetConverter is not None

    def test_convert_from_nemo_file(self, tmp_path):
        nemo_model = os.environ.get("PARAKEET_NEMO_MODEL", "")
        if not nemo_model or not os.path.isfile(nemo_model):
            pytest.skip("PARAKEET_NEMO_MODEL is not set or file does not exist")

        from ctranslate2.converters.nemo_parakeet import NemoParakeetConverter

        output_dir = str(tmp_path / "parakeet_ct2")
        converter = NemoParakeetConverter(nemo_model)
        converter.convert(output_dir, quantization="float32", force=True)

        assert os.path.isfile(os.path.join(output_dir, "model.bin"))
        assert os.path.isfile(os.path.join(output_dir, "config.json"))
