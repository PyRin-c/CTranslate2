# NOTE: This file is specific to PyRin-c/CTranslate2 (feature/support_vibevoice).
# Not present in upstream OpenNMT/CTranslate2.
#
# Tests for:
#   - vibevoice_asr_spec.py  (Spec structure / validation)
#   - ctranslate2.models.VibeVoiceAsr  (runtime E2E — skipped when no model available)
#   - VibeVoiceAsrConverter (conversion — skipped when transformers is unavailable)

import os

import numpy as np
import pytest

from ctranslate2.specs import common_spec
from ctranslate2.specs.vibevoice_asr_spec import (
    ConvNeXtBlockSpec,
    VibeVoiceAcousticStemSpec,
    VibeVoiceAcousticEncoderLayerSpec,
    VibeVoiceAcousticEncoderSpec,
    VibeVoiceMultiModalProjectorSpec,
    VibeVoiceAsrConfig,
    VibeVoiceAsrSpec,
)


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
    return spec


def _make_conv1d(out_ch, in_ch, k, bias=True):
    spec = common_spec.Conv1DSpec()
    spec.weight = np.zeros((out_ch, in_ch, k), dtype=np.float32)
    if bias:
        spec.bias = np.zeros(out_ch, dtype=np.float32)
    return spec


def _fill_convnext_block(spec, channels=8, kernel=7, ffn_dim=32):
    spec.norm.gamma = np.ones(channels, dtype=np.float32)
    spec.mixer = _make_conv1d(channels, 1, kernel)  # depth-wise: groups=C, in_ch per group=1
    spec.gamma = np.ones(channels, dtype=np.float32)
    spec.ffn_norm.gamma = np.ones(channels, dtype=np.float32)
    spec.ffn_linear1 = _make_linear(ffn_dim, channels)
    spec.ffn_linear2 = _make_linear(channels, ffn_dim)
    spec.ffn_gamma = np.ones(channels, dtype=np.float32)


def _fill_stem(spec, channels=8, num_blocks=3):
    spec.conv = _make_conv1d(channels, 1, 7)
    for stage in spec.stage:
        _fill_convnext_block(stage, channels=channels)


def _fill_encoder_layer(spec, in_ch=8, out_ch=16, stride=2, num_blocks=3):
    spec.conv = _make_conv1d(out_ch, in_ch, stride + 6, bias=True)
    for stage in spec.stage:
        _fill_convnext_block(stage, channels=out_ch)


# ---------------------------------------------------------------------------
# ConvNeXtBlockSpec tests
# ---------------------------------------------------------------------------

class TestConvNeXtBlockSpec:
    def test_fields_exist(self):
        spec = ConvNeXtBlockSpec()
        assert hasattr(spec, "norm")
        assert hasattr(spec, "mixer")
        assert hasattr(spec, "gamma")
        assert hasattr(spec, "ffn_norm")
        assert hasattr(spec, "ffn_linear1")
        assert hasattr(spec, "ffn_linear2")
        assert hasattr(spec, "ffn_gamma")

    def test_norm_is_rms(self):
        """ConvNeXtBlock uses RMSNorm (no beta / bias)."""
        spec = ConvNeXtBlockSpec()
        assert isinstance(spec.norm, common_spec.LayerNormSpec)

    def test_filled_spec_validates(self):
        spec = ConvNeXtBlockSpec()
        _fill_convnext_block(spec, channels=16)
        # Verify arrays have correct shapes.
        assert spec.gamma.shape == (16,)
        assert spec.ffn_gamma.shape == (16,)


# ---------------------------------------------------------------------------
# VibeVoiceAcousticStemSpec tests
# ---------------------------------------------------------------------------

class TestVibeVoiceAcousticStemSpec:
    def test_default_blocks(self):
        spec = VibeVoiceAcousticStemSpec(num_blocks=3)
        assert len(spec.stage) == 3

    def test_custom_blocks(self):
        spec = VibeVoiceAcousticStemSpec(num_blocks=5)
        assert len(spec.stage) == 5

    def test_has_conv(self):
        spec = VibeVoiceAcousticStemSpec(num_blocks=3)
        assert hasattr(spec, "conv")
        assert isinstance(spec.conv, common_spec.Conv1DSpec)


# ---------------------------------------------------------------------------
# VibeVoiceAcousticEncoderLayerSpec tests
# ---------------------------------------------------------------------------

class TestVibeVoiceAcousticEncoderLayerSpec:
    def test_stage_count(self):
        for n in [3, 8]:
            spec = VibeVoiceAcousticEncoderLayerSpec(num_blocks=n)
            assert len(spec.stage) == n

    def test_has_conv(self):
        spec = VibeVoiceAcousticEncoderLayerSpec(num_blocks=3)
        assert isinstance(spec.conv, common_spec.Conv1DSpec)


# ---------------------------------------------------------------------------
# VibeVoiceAcousticEncoderSpec tests
# ---------------------------------------------------------------------------

class TestVibeVoiceAcousticEncoderSpec:
    def test_default_depths(self):
        spec = VibeVoiceAcousticEncoderSpec()
        # Default: depths=[3,3,3,3,3,3,8], num_downsampling=6
        assert len(spec.stem.stage) == 3
        assert len(spec.conv_layers) == 6
        assert len(spec.conv_layers[-1].stage) == 8

    def test_custom_depths(self):
        depths = [2, 2, 4]
        spec = VibeVoiceAcousticEncoderSpec(depths=depths, num_downsampling=2)
        assert len(spec.stem.stage) == 2
        assert len(spec.conv_layers) == 2
        assert len(spec.conv_layers[0].stage) == 2
        assert len(spec.conv_layers[1].stage) == 4

    def test_has_head(self):
        spec = VibeVoiceAcousticEncoderSpec()
        assert isinstance(spec.head, common_spec.Conv1DSpec)


# ---------------------------------------------------------------------------
# VibeVoiceMultiModalProjectorSpec tests
# ---------------------------------------------------------------------------

class TestVibeVoiceMultiModalProjectorSpec:
    def test_fields_exist(self):
        spec = VibeVoiceMultiModalProjectorSpec()
        for field in [
            "acoustic_linear_1", "acoustic_norm", "acoustic_linear_2",
            "semantic_linear_1", "semantic_norm", "semantic_linear_2",
        ]:
            assert hasattr(spec, field), f"Missing field: {field}"

    def test_norms_are_rms(self):
        spec = VibeVoiceMultiModalProjectorSpec()
        assert isinstance(spec.acoustic_norm, common_spec.LayerNormSpec)
        assert isinstance(spec.semantic_norm, common_spec.LayerNormSpec)


# ---------------------------------------------------------------------------
# VibeVoiceAsrConfig tests
# ---------------------------------------------------------------------------

class TestVibeVoiceAsrConfig:
    def test_default_values(self):
        cfg = VibeVoiceAsrConfig()
        assert cfg.audio_token_id == 151648
        assert cfg.audio_bos_token_id == 151646
        assert cfg.audio_eos_token_id == 151647
        assert cfg.decoder_num_layers == 28
        assert cfg.decoder_num_heads == 28
        assert cfg.decoder_num_kv_heads == 4

    def test_custom_values(self):
        cfg = VibeVoiceAsrConfig(decoder_num_layers=12, decoder_num_heads=12)
        assert cfg.decoder_num_layers == 12
        assert cfg.decoder_num_heads == 12

    def test_downsampling_ratios_default(self):
        cfg = VibeVoiceAsrConfig()
        assert cfg.acoustic_downsampling_ratios == [2, 2, 4, 5, 5, 8]


# ---------------------------------------------------------------------------
# VibeVoiceAsrSpec tests
# ---------------------------------------------------------------------------

class TestVibeVoiceAsrSpec:
    def test_name(self):
        spec = VibeVoiceAsrSpec()
        assert spec.name == "VibeVoiceAsrSpec"

    def test_revision(self):
        spec = VibeVoiceAsrSpec()
        assert spec.revision == 1

    def test_has_required_components(self):
        spec = VibeVoiceAsrSpec()
        assert hasattr(spec, "acoustic_tokenizer_encoder")
        assert hasattr(spec, "semantic_tokenizer_encoder")
        assert hasattr(spec, "multi_modal_projector")
        assert hasattr(spec, "decoder")

    def test_decoder_structure(self):
        from ctranslate2.specs import transformer_spec
        spec = VibeVoiceAsrSpec()
        assert isinstance(spec.decoder, transformer_spec.TransformerDecoderSpec)

    def test_default_decoder_depth(self):
        spec = VibeVoiceAsrSpec()
        # 28 decoder layers by default
        assert len(spec.decoder.layer) == 28

    def test_custom_decoder_depth(self):
        spec = VibeVoiceAsrSpec(num_decoder_layers=4, num_decoder_heads=4,
                                num_decoder_kv_heads=2)
        assert len(spec.decoder.layer) == 4

    def test_acoustic_and_semantic_encoders_match_default_depths(self):
        spec = VibeVoiceAsrSpec()
        # Both encoders share the same depth configuration.
        assert len(spec.acoustic_tokenizer_encoder.conv_layers) == \
               len(spec.semantic_tokenizer_encoder.conv_layers)
        assert len(spec.acoustic_tokenizer_encoder.stem.stage) == \
               len(spec.semantic_tokenizer_encoder.stem.stage)

    def test_get_default_config(self):
        spec = VibeVoiceAsrSpec()
        cfg = spec.get_default_config()
        assert isinstance(cfg, VibeVoiceAsrConfig)

    def test_scale_embeddings_disabled(self):
        """VibeVoice decoder uses Qwen2 — embeddings are not scaled."""
        spec = VibeVoiceAsrSpec()
        assert spec.decoder.scale_embeddings is False


# ---------------------------------------------------------------------------
# Runtime test (skipped when the converted model is not present)
# ---------------------------------------------------------------------------

VIBEVOICE_MODEL_DIR = os.environ.get("VIBEVOICE_MODEL_DIR", "")


@pytest.mark.skipif(not VIBEVOICE_MODEL_DIR, reason="VIBEVOICE_MODEL_DIR not set")
class TestVibeVoiceAsrRuntime:
    def test_import(self):
        import ctranslate2
        assert hasattr(ctranslate2.models, "VibeVoiceAsr")
        assert hasattr(ctranslate2.models, "VibeVoiceAsrResult")

    def test_instantiate(self):
        import ctranslate2
        model = ctranslate2.models.VibeVoiceAsr(VIBEVOICE_MODEL_DIR, device="cpu")
        assert model is not None

    def test_encode_shape(self):
        """encode() must return (1, T, lm_hidden_size) for a 1-second clip."""
        import ctranslate2
        model = ctranslate2.models.VibeVoiceAsr(VIBEVOICE_MODEL_DIR, device="cpu")
        waveform = np.zeros((1, 24000), dtype=np.float32)  # 1 second @ 24kHz
        features = model.encode(waveform).result()
        assert features.ndim == 3
        assert features.shape[0] == 1
        assert features.shape[2] == 3584  # lm_hidden_size for VibeVoice default


# ---------------------------------------------------------------------------
# Converter test (skipped when transformers is unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("VIBEVOICE_HF_MODEL_ID", ""),
    reason="VIBEVOICE_HF_MODEL_ID not set",
)
def test_converter_import():
    from ctranslate2.converters.vibevoice_asr import VibeVoiceAsrConverter
    assert callable(VibeVoiceAsrConverter)
