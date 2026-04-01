"""Specification for VibeVoice-ASR model.

Architecture (verified from Transformers v5.4.0 source):

  VibeVoiceAsrForConditionalGeneration
    .acoustic_tokenizer_encoder  (VibeVoiceAcousticTokenizerEncoderModel)
        .stem
            .conv          CausalConv1d (1 → num_filters=32, kernel=7)
            .stage[0..2]   ConvNeXtBlock × depths[0]=3
        .conv_layers[0..5] (6 downsampling layers)
            .conv          CausalConv1d (strided, ratio=[2,2,4,5,5,8][i])
            .stage[0..N]   ConvNeXtBlock × depths[i+1]
        .head              CausalConv1d (channels → hidden_size=64)
    .semantic_tokenizer_encoder  (同構造, hidden_size=128)
    .multi_modal_projector  (acoustic+semantic → lm_hidden_size=3584)
        .acoustic_linear_1, .acoustic_norm, .acoustic_linear_2
        .semantic_linear_1, .semantic_norm, .semantic_linear_2
    .language_model  (Qwen2ForCausalLM)

ConvNeXtBlock forward (VibeVoiceAcousticTokenizerConvNext1dLayer):
  # Sub-block 1: mixer path
  h = norm(x)
  h = mixer(h)       # depth-wise causal conv1d (groups=C)
  h = gamma * h      # LayerScale (nn.Parameter)
  x = x + h

  # Sub-block 2: FFN path
  h = ffn_norm(x)
  h = linear2(gelu(linear1(h)))
  h = ffn_gamma * h  # LayerScale (nn.Parameter)
  x = x + h

CausalConv1d wraps a plain nn.Conv1d at .conv; converter reads .conv.weight/.bias.
"""

import numpy as np

from ctranslate2.specs import common_spec, model_spec, transformer_spec


# ---------------------------------------------------------------------------
# ConvNeXt block (two-path: mixer + FFN, each with pre-norm + LayerScale)
# ---------------------------------------------------------------------------

class ConvNeXtBlockSpec(model_spec.LayerSpec):
    """VibeVoiceAcousticTokenizerConvNext1dLayer の CTranslate2 スペック。

    Weight paths (prefix は呼び出し元が付ける):
      norm/gamma
      mixer/weight, mixer/bias
      gamma                  (LayerScale, shape [C])
      ffn_norm/gamma
      ffn_linear1/weight, ffn_linear1/bias
      ffn_linear2/weight, ffn_linear2/bias
      ffn_gamma              (LayerScale, shape [C])
    """

    def __init__(self):
        # Sub-block 1: pre-norm → depth-wise causal conv → LayerScale
        self.norm = common_spec.LayerNormSpec(rms_norm=True)  # .weight
        self.mixer = common_spec.Conv1DSpec()                  # actual nn.Conv1d weights
        self.gamma = None                                       # LayerScale nn.Parameter [C]

        # Sub-block 2: pre-norm → FFN → LayerScale
        self.ffn_norm = common_spec.LayerNormSpec(rms_norm=True)
        self.ffn_linear1 = common_spec.LinearSpec()
        self.ffn_linear2 = common_spec.LinearSpec()
        self.ffn_gamma = None                                   # LayerScale nn.Parameter [C]


# ---------------------------------------------------------------------------
# Stem: initial conv + ConvNeXt blocks (depths[0])
# ---------------------------------------------------------------------------

class VibeVoiceAcousticStemSpec(model_spec.LayerSpec):
    """EncoderStem = initial CausalConv1d + depths[0] ConvNeXt blocks."""

    def __init__(self, num_blocks: int):
        self.conv = common_spec.Conv1DSpec()                       # CausalConv1d.conv
        self.stage = [ConvNeXtBlockSpec() for _ in range(num_blocks)]


# ---------------------------------------------------------------------------
# EncoderLayer: downsampling conv + ConvNeXt blocks (depths[i+1])
# ---------------------------------------------------------------------------

class VibeVoiceAcousticEncoderLayerSpec(model_spec.LayerSpec):
    """EncoderLayer i = downsampling CausalConv1d + depths[i+1] ConvNeXt blocks."""

    def __init__(self, num_blocks: int):
        self.conv = common_spec.Conv1DSpec()                       # CausalConv1d.conv
        self.stage = [ConvNeXtBlockSpec() for _ in range(num_blocks)]


# ---------------------------------------------------------------------------
# Full encoder (stem + conv_layers + head)
# ---------------------------------------------------------------------------

class VibeVoiceAcousticEncoderSpec(model_spec.LayerSpec):
    """VibeVoiceAcousticTokenizerEncoderModel の CTranslate2 スペック。

    depths=[3,3,3,3,3,3,8] に対して:
      stem         : ConvNeXt × depths[0]  (= 3 blocks)
      conv_layers_0: downsampling + ConvNeXt × depths[1]  (= 3 blocks)
      ...
      conv_layers_5: downsampling + ConvNeXt × depths[6]  (= 8 blocks)
      head         : projection conv → hidden_size (64 or 128)
    """

    def __init__(
        self,
        depths: list = None,
        num_downsampling: int = 6,
    ):
        if depths is None:
            depths = [3, 3, 3, 3, 3, 3, 8]

        self.stem = VibeVoiceAcousticStemSpec(depths[0])
        self.conv_layers = [
            VibeVoiceAcousticEncoderLayerSpec(depths[i + 1])
            for i in range(num_downsampling)
        ]
        self.head = common_spec.Conv1DSpec()  # CausalConv1d.conv → hidden_size


# ---------------------------------------------------------------------------
# Multi-modal projector (acoustic + semantic → lm_hidden_size)
# ---------------------------------------------------------------------------

class VibeVoiceMultiModalProjectorSpec(model_spec.LayerSpec):
    """VibeVoiceAsrMultiModalProjector の CTranslate2 スペック。

    acoustic: linear_1 → RMSNorm → linear_2
    semantic: linear_1 → RMSNorm → linear_2
    (両者の出力を加算して lm_hidden_size の音声埋め込みを得る)
    """

    def __init__(self):
        self.acoustic_linear_1 = common_spec.LinearSpec()
        self.acoustic_norm = common_spec.LayerNormSpec(rms_norm=True)
        self.acoustic_linear_2 = common_spec.LinearSpec()
        self.semantic_linear_1 = common_spec.LinearSpec()
        self.semantic_norm = common_spec.LayerNormSpec(rms_norm=True)
        self.semantic_linear_2 = common_spec.LinearSpec()


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class VibeVoiceAsrConfig(model_spec.ModelConfig):
    """config.json に書き出すパラメータ（C++ ランタイムが参照）。"""

    def __init__(
        self,
        audio_token_id: int = 151648,
        audio_bos_token_id: int = 151646,
        audio_eos_token_id: int = 151647,
        acoustic_tokenizer_chunk_size: int = 1440000,
        acoustic_hidden_size: int = 64,
        semantic_hidden_size: int = 128,
        acoustic_downsampling_ratios: list = None,
        acoustic_depths: list = None,
        acoustic_num_filters: int = 32,
        acoustic_kernel_size: int = 7,
        acoustic_vae_std: float = 0.625,
        decoder_hidden_size: int = 3584,
        decoder_num_layers: int = 28,
        decoder_num_heads: int = 28,
        decoder_num_kv_heads: int = 4,
        decoder_intermediate_size: int = 18944,
    ):
        if acoustic_downsampling_ratios is None:
            acoustic_downsampling_ratios = [2, 2, 4, 5, 5, 8]
        if acoustic_depths is None:
            acoustic_depths = [3, 3, 3, 3, 3, 3, 8]

        super().__init__(
            audio_token_id=audio_token_id,
            audio_bos_token_id=audio_bos_token_id,
            audio_eos_token_id=audio_eos_token_id,
            acoustic_tokenizer_chunk_size=acoustic_tokenizer_chunk_size,
            acoustic_hidden_size=acoustic_hidden_size,
            semantic_hidden_size=semantic_hidden_size,
            acoustic_downsampling_ratios=acoustic_downsampling_ratios,
            acoustic_depths=acoustic_depths,
            acoustic_num_filters=acoustic_num_filters,
            acoustic_kernel_size=acoustic_kernel_size,
            acoustic_vae_std=acoustic_vae_std,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_layers=decoder_num_layers,
            decoder_num_heads=decoder_num_heads,
            decoder_num_kv_heads=decoder_num_kv_heads,
            decoder_intermediate_size=decoder_intermediate_size,
        )


# ---------------------------------------------------------------------------
# Top-level model spec
# ---------------------------------------------------------------------------

class VibeVoiceAsrSpec(model_spec.LanguageModelSpec):
    """VibeVoice-ASR 全体のスペック。

    Weight path 例（model.bin 内）:
      acoustic_tokenizer_encoder/stem/conv/weight
      acoustic_tokenizer_encoder/stem/stage_0/norm/gamma
      acoustic_tokenizer_encoder/stem/stage_0/mixer/weight
      acoustic_tokenizer_encoder/stem/stage_0/gamma
      acoustic_tokenizer_encoder/stem/stage_0/ffn_norm/gamma
      acoustic_tokenizer_encoder/stem/stage_0/ffn_linear1/weight
      acoustic_tokenizer_encoder/stem/stage_0/ffn_linear2/weight
      acoustic_tokenizer_encoder/stem/stage_0/ffn_gamma
      acoustic_tokenizer_encoder/conv_layers_0/conv/weight
      acoustic_tokenizer_encoder/conv_layers_0/stage_0/...
      acoustic_tokenizer_encoder/head/weight
      semantic_tokenizer_encoder/...
      multi_modal_projector/acoustic_linear_1/weight
      multi_modal_projector/acoustic_norm/gamma
      multi_modal_projector/acoustic_linear_2/weight
      multi_modal_projector/semantic_linear_1/weight
      multi_modal_projector/semantic_norm/gamma
      multi_modal_projector/semantic_linear_2/weight
      decoder/embeddings/weight
      decoder/layer_0/...    (Qwen2 28 layers)
      decoder/layer_norm/gamma
      decoder/projection/weight
    """

    _ACOUSTIC_DEPTHS = [3, 3, 3, 3, 3, 3, 8]
    _NUM_DOWNSAMPLING = 6

    def __init__(
        self,
        num_decoder_layers: int = 28,
        num_decoder_heads: int = 28,
        num_decoder_kv_heads: int = 4,
        acoustic_depths: list = None,
        num_downsampling: int = 6,
    ):
        super().__init__()
        if acoustic_depths is None:
            acoustic_depths = self._ACOUSTIC_DEPTHS

        self.acoustic_tokenizer_encoder = VibeVoiceAcousticEncoderSpec(
            depths=acoustic_depths,
            num_downsampling=num_downsampling,
        )
        self.semantic_tokenizer_encoder = VibeVoiceAcousticEncoderSpec(
            depths=acoustic_depths,
            num_downsampling=num_downsampling,
        )
        self.multi_modal_projector = VibeVoiceMultiModalProjectorSpec()
        self.decoder = transformer_spec.TransformerDecoderSpec(
            num_decoder_layers,
            num_decoder_heads,
            pre_norm=True,
            activation=common_spec.Activation.SWISH,
            with_encoder_attention=False,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
            num_heads_kv=num_decoder_kv_heads,
        )
        self.decoder.scale_embeddings = False

    @property
    def name(self):
        return "VibeVoiceAsrSpec"

    @property
    def revision(self):
        return 1

    def get_default_config(self):
        return VibeVoiceAsrConfig()

    def get_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]
