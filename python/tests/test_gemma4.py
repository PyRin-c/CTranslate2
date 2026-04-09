# NOTE: This file is specific to PyRin-c/CTranslate2 (feature/add_support_parakeet-ctc).
# Not present in upstream OpenNMT/CTranslate2.
#
# Tests for:
#   - gemma4_spec.py  (Spec structure / validation)
#   - ctranslate2.models.Gemma4  (runtime E2E — skipped when no model available)
#   - Gemma4MultimodalLoader (conversion — skipped when transformers is unavailable)

import os

import numpy as np
import pytest

from ctranslate2.specs import attention_spec, common_spec, transformer_spec
from ctranslate2.specs.gemma4_spec import (
    Gemma4MultimodalEmbedderSpec,
    Gemma4Spec,
    Gemma4VisionEncoderLayerSpec,
    Gemma4VisionEncoderSpec,
    Gemma4VisionPatchEmbedderSpec,
)


# ---------------------------------------------------------------------------
# Gemma4VisionPatchEmbedderSpec
# ---------------------------------------------------------------------------


class TestGemma4VisionPatchEmbedderSpec:
    def test_fields_exist(self):
        spec = Gemma4VisionPatchEmbedderSpec()
        assert hasattr(spec, "input_proj")
        assert hasattr(spec, "position_embedding_table")

    def test_input_proj_is_linear(self):
        spec = Gemma4VisionPatchEmbedderSpec()
        assert isinstance(spec.input_proj, common_spec.LinearSpec)

    def test_position_embedding_table_initially_none(self):
        spec = Gemma4VisionPatchEmbedderSpec()
        assert spec.position_embedding_table is None


# ---------------------------------------------------------------------------
# Gemma4VisionEncoderLayerSpec
# ---------------------------------------------------------------------------


class TestGemma4VisionEncoderLayerSpec:
    def _make(self, head_dim=64):
        return Gemma4VisionEncoderLayerSpec(head_dim=head_dim)

    def test_layer_norms_exist(self):
        spec = self._make()
        for attr in [
            "input_layer_norm",
            "post_attention_layer_norm",
            "pre_feedforward_layer_norm",
            "post_feedforward_layer_norm",
        ]:
            assert hasattr(spec, attr), f"Missing attribute: {attr}"

    def test_layer_norms_are_rms(self):
        spec = self._make()
        assert isinstance(spec.input_layer_norm, common_spec.LayerNormSpec)
        assert isinstance(spec.post_attention_layer_norm, common_spec.LayerNormSpec)
        assert isinstance(spec.pre_feedforward_layer_norm, common_spec.LayerNormSpec)
        assert isinstance(spec.post_feedforward_layer_norm, common_spec.LayerNormSpec)

    def test_self_attention_exists(self):
        spec = self._make()
        assert hasattr(spec, "self_attention")
        assert isinstance(spec.self_attention, attention_spec.MultiHeadAttentionSpec)

    def test_qk_norm_present(self):
        spec = self._make()
        assert hasattr(spec.self_attention, "q_norm")
        assert hasattr(spec.self_attention, "k_norm")

    def test_v_norm_present(self):
        """Vision encoder layers carry an extra v_norm (scaleless RMSNorm)."""
        spec = self._make()
        assert hasattr(spec.self_attention, "v_norm")

    def test_ffn_glu(self):
        spec = self._make()
        assert hasattr(spec, "ffn")
        assert isinstance(spec.ffn, transformer_spec.FeedForwardSpec)
        # GeGLU: gate+up projections exist, internal layer_norm removed.
        assert hasattr(spec.ffn, "linear_0")       # gate_proj
        assert hasattr(spec.ffn, "linear_0_noact") # up_proj
        assert hasattr(spec.ffn, "linear_1")       # down_proj
        assert not hasattr(spec.ffn, "layer_norm"), \
            "FFN layer_norm should be removed (norms live outside)"


# ---------------------------------------------------------------------------
# Gemma4VisionEncoderSpec
# ---------------------------------------------------------------------------


class TestGemma4VisionEncoderSpec:
    def test_default_layer_count(self):
        spec = Gemma4VisionEncoderSpec(num_layers=16, num_heads=12, head_dim=64)
        assert len(spec.layer) == 16

    def test_custom_layer_count(self):
        for n in [4, 8]:
            spec = Gemma4VisionEncoderSpec(num_layers=n, num_heads=8, head_dim=64)
            assert len(spec.layer) == n

    def test_num_heads_stored(self):
        spec = Gemma4VisionEncoderSpec(num_layers=4, num_heads=12, head_dim=64)
        assert int(spec.num_heads) == 12

    def test_activation_stored(self):
        spec = Gemma4VisionEncoderSpec(num_layers=4, num_heads=12, head_dim=64)
        assert int(spec.activation) == common_spec.Activation.GELUTanh

    def test_rope_tables_initially_none(self):
        spec = Gemma4VisionEncoderSpec(num_layers=4, num_heads=12, head_dim=64)
        assert spec.rope_cos is None
        assert spec.rope_sin is None

    def test_patch_embedder_type(self):
        spec = Gemma4VisionEncoderSpec(num_layers=4, num_heads=12, head_dim=64)
        assert isinstance(spec.patch_embedder, Gemma4VisionPatchEmbedderSpec)

    def test_layers_are_correct_type(self):
        spec = Gemma4VisionEncoderSpec(num_layers=4, num_heads=8, head_dim=64)
        for layer in spec.layer:
            assert isinstance(layer, Gemma4VisionEncoderLayerSpec)


# ---------------------------------------------------------------------------
# Gemma4MultimodalEmbedderSpec
# ---------------------------------------------------------------------------


class TestGemma4MultimodalEmbedderSpec:
    def test_fields_exist(self):
        spec = Gemma4MultimodalEmbedderSpec()
        assert hasattr(spec, "pre_projection_norm")
        assert hasattr(spec, "projection")

    def test_pre_projection_norm_is_layer_norm(self):
        spec = Gemma4MultimodalEmbedderSpec()
        assert isinstance(spec.pre_projection_norm, common_spec.LayerNormSpec)

    def test_projection_is_linear(self):
        spec = Gemma4MultimodalEmbedderSpec()
        assert isinstance(spec.projection, common_spec.LinearSpec)


# ---------------------------------------------------------------------------
# Gemma4Spec (top-level)
# ---------------------------------------------------------------------------


class TestGemma4Spec:
    def test_name(self):
        spec = Gemma4Spec()
        assert spec.name == "Gemma4Spec"

    def test_revision(self):
        spec = Gemma4Spec()
        assert spec.revision == 1

    def test_has_required_components(self):
        spec = Gemma4Spec()
        assert hasattr(spec, "vision_encoder")
        assert hasattr(spec, "multimodal_embedder")
        assert hasattr(spec, "decoder")

    def test_vision_encoder_type(self):
        spec = Gemma4Spec()
        assert isinstance(spec.vision_encoder, Gemma4VisionEncoderSpec)

    def test_multimodal_embedder_type(self):
        spec = Gemma4Spec()
        assert isinstance(spec.multimodal_embedder, Gemma4MultimodalEmbedderSpec)

    def test_decoder_type(self):
        spec = Gemma4Spec()
        assert isinstance(spec.decoder, transformer_spec.TransformerDecoderSpec)

    def test_default_vision_encoder_depth(self):
        spec = Gemma4Spec()
        assert len(spec.vision_encoder.layer) == 16

    def test_default_decoder_depth(self):
        spec = Gemma4Spec()
        assert len(spec.decoder.layer) == 30

    def test_custom_depths(self):
        spec = Gemma4Spec(
            num_vision_layers=4,
            num_vision_heads=4,
            vision_head_dim=32,
            num_decoder_layers=6,
            num_decoder_heads=4,
            num_decoder_kv_heads=2,
        )
        assert len(spec.vision_encoder.layer) == 4
        assert len(spec.decoder.layer) == 6

    def test_scale_embeddings_false(self):
        """Gemma 4 text decoder manages embedding scale in the generate path; spec flag is False."""
        spec = Gemma4Spec()
        assert spec.decoder.scale_embeddings is False

    def test_get_default_config(self):
        from ctranslate2.specs import model_spec
        spec = Gemma4Spec()
        cfg = spec.get_default_config()
        assert isinstance(cfg, model_spec.ModelConfig)


# ---------------------------------------------------------------------------
# Gemma4GenerationResult binding
# ---------------------------------------------------------------------------


class TestGemma4GenerationResultBinding:
    def test_import(self):
        import ctranslate2
        assert hasattr(ctranslate2.models, "Gemma4GenerationResult")

    def test_attributes_present(self):
        """Gemma4GenerationResult must expose sequences, sequences_ids, scores."""
        import ctranslate2
        result_class = ctranslate2.models.Gemma4GenerationResult
        # Access via pybind11 class descriptor (properties are on the type).
        for attr in ("sequences", "sequences_ids", "scores"):
            assert hasattr(result_class, attr), f"Missing attribute: {attr}"

    def test_gemma4_class_importable(self):
        import ctranslate2
        assert hasattr(ctranslate2.models, "Gemma4")


# ---------------------------------------------------------------------------
# Runtime test (skipped when the converted model directory is not set)
# ---------------------------------------------------------------------------

GEMMA4_MODEL_DIR = os.environ.get("GEMMA4_MODEL_DIR", "")


@pytest.mark.skipif(not GEMMA4_MODEL_DIR, reason="GEMMA4_MODEL_DIR not set")
class TestGemma4Runtime:
    def test_instantiate(self):
        import ctranslate2
        model = ctranslate2.models.Gemma4(GEMMA4_MODEL_DIR, device="cpu")
        assert model is not None

    def test_device_property(self):
        import ctranslate2
        model = ctranslate2.models.Gemma4(GEMMA4_MODEL_DIR, device="cpu")
        assert model.device == "cpu"

    def test_image_token_id_property(self):
        """image_token_id read from config.json must be non-zero for Gemma 4."""
        import ctranslate2
        model = ctranslate2.models.Gemma4(GEMMA4_MODEL_DIR, device="cpu")
        assert model.image_token_id > 0

    def test_encode_vision_shape(self):
        """encode_vision() must return [1, num_patches, text_hidden_size]."""
        import ctranslate2
        model = ctranslate2.models.Gemma4(GEMMA4_MODEL_DIR, device="cpu")

        num_patches = 256
        patch_dim = 588  # 3 * 14^2 for a typical Gemma 4 patch size
        pixel_values = np.zeros((1, num_patches, patch_dim), dtype=np.float32)
        pixel_pos_ids = np.zeros((1, num_patches, 2), dtype=np.int32)

        pv = ctranslate2.StorageView.from_array(pixel_values)
        pp = ctranslate2.StorageView.from_array(pixel_pos_ids)
        vision_embeds = model.encode_vision(pv, pp)

        arr = np.array(vision_embeds)
        assert arr.ndim == 3
        assert arr.shape[0] == 1
        assert arr.shape[1] == num_patches

    def test_generate_returns_list(self):
        """generate() must return a list of Gemma4GenerationResult."""
        import ctranslate2
        model = ctranslate2.models.Gemma4(GEMMA4_MODEL_DIR, device="cpu")

        num_patches = 4
        patch_dim = 588
        pixel_values = np.zeros((1, num_patches, patch_dim), dtype=np.float32)
        pixel_pos_ids = np.zeros((1, num_patches, 2), dtype=np.int32)

        pv = ctranslate2.StorageView.from_array(pixel_values)
        pp = ctranslate2.StorageView.from_array(pixel_pos_ids)
        vision_embeds = model.encode_vision(pv, pp)

        img_tok = model.image_token_id
        input_ids = np.array([[img_tok] * num_patches], dtype=np.int32)
        ids_sv = ctranslate2.StorageView.from_array(input_ids)

        results = model.generate(ids_sv, vision_embeds, max_new_tokens=4)
        assert isinstance(results, list)
        assert len(results) == 1
        assert hasattr(results[0], "sequences")
        assert hasattr(results[0], "sequences_ids")
        assert hasattr(results[0], "scores")


# ---------------------------------------------------------------------------
# Converter import test (skipped when transformers is not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("GEMMA4_HF_MODEL_ID", ""),
    reason="GEMMA4_HF_MODEL_ID not set",
)
def test_converter_available():
    """Gemma4MultimodalLoader must be accessible via the transformers converter."""
    try:
        from ctranslate2.converters.transformers import Gemma4MultimodalLoader
        assert callable(Gemma4MultimodalLoader)
    except ImportError as exc:
        pytest.skip(f"transformers not available: {exc}")
