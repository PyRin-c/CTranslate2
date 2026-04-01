"""Converter for microsoft/VibeVoice-ASR-HF to CTranslate2 format.

Architecture (Transformers v5.4.0):
  VibeVoiceAsrForConditionalGeneration
    ├── acoustic_tokenizer_encoder  (VibeVoiceAcousticTokenizerEncoderModel)
    │     ├── stem
    │     │     ├── conv     VibeVoiceAcousticTokenizerCausalConv1d (.conv → nn.Conv1d)
    │     │     └── stage[0..depths[0]-1]  VibeVoiceAcousticTokenizerConvNext1dLayer
    │     ├── conv_layers[0..5]
    │     │     ├── conv     VibeVoiceAcousticTokenizerCausalConv1d (strided)
    │     │     └── stage[0..depths[i+1]-1]  ConvNext1dLayer
    │     └── head            VibeVoiceAcousticTokenizerCausalConv1d
    ├── semantic_tokenizer_encoder  (同構造、hidden_size=128)
    ├── multi_modal_projector  (acoustic+semantic → lm_hidden_size)
    │     ├── acoustic_linear_1, acoustic_norm, acoustic_linear_2
    │     └── semantic_linear_1, semantic_norm, semantic_linear_2
    └── language_model  (Qwen2ForCausalLM)

ConvNext1dLayer 内部 (VibeVoiceAcousticTokenizerConvNext1dLayer):
  .norm      RMSNorm
  .mixer     CausalConv1d (.conv → depth-wise nn.Conv1d)
  .gamma     nn.Parameter [C]   (LayerScale)
  .ffn_norm  RMSNorm
  .ffn.linear1  nn.Linear
  .ffn.linear2  nn.Linear
  .ffn_gamma nn.Parameter [C]   (FFN LayerScale)

Usage:
  ct2-vibevoice-asr-converter \\
      --model microsoft/VibeVoice-ASR-HF \\
      --output_dir vibevoice-asr-ct2 \\
      --quantization bfloat16 \\
      --copy_files tokenizer.json preprocessor_config.json
"""

import argparse
import gc
import os

from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec
from ctranslate2.specs.vibevoice_asr_spec import VibeVoiceAsrConfig, VibeVoiceAsrSpec


class VibeVoiceAsrConverter(Converter):
    """Converts microsoft/VibeVoice-ASR-HF to CTranslate2 format."""

    def __init__(
        self,
        model_name_or_path: str,
        low_cpu_mem_usage: bool = True,
        copy_files: list = None,
    ):
        """
        Args:
          model_name_or_path: HuggingFace model ID または ローカルパス。
          low_cpu_mem_usage: True にすると CPU メモリを節約してロードする。
          copy_files: 変換先ディレクトリにコピーする追加ファイル名リスト
                      (例: ["tokenizer.json", "preprocessor_config.json"])。
        """
        self._model_path = model_name_or_path
        self._low_cpu_mem_usage = low_cpu_mem_usage
        self._copy_files = copy_files or []

    # ------------------------------------------------------------------
    # Converter interface
    # ------------------------------------------------------------------

    def _load(self):
        import torch
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        print(f"Loading {self._model_path} ...")
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self._model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=self._low_cpu_mem_usage,
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(self._model_path)
        tokenizer = processor.tokenizer

        spec = self._build_spec(model)
        self._set_config(spec, model)
        self._set_vocabulary(spec, tokenizer, vocab_size=model.config.text_config.vocab_size)
        self._register_copy_files(spec, processor)

        return spec

    # ------------------------------------------------------------------
    # Spec 構築
    # ------------------------------------------------------------------

    def _build_spec(self, model):
        cfg = model.config
        enc_cfg = cfg.acoustic_tokenizer_encoder_config

        depths = list(enc_cfg.depths)
        num_downsampling = len(enc_cfg.downsampling_ratios)

        lm_cfg = cfg.text_config
        num_layers = lm_cfg.num_hidden_layers
        num_heads = lm_cfg.num_attention_heads
        num_kv_heads = lm_cfg.num_key_value_heads

        spec = VibeVoiceAsrSpec(
            num_decoder_layers=num_layers,
            num_decoder_heads=num_heads,
            num_decoder_kv_heads=num_kv_heads,
            acoustic_depths=depths,
            num_downsampling=num_downsampling,
        )

        print("Setting acoustic tokenizer encoder weights ...")
        self._set_acoustic_encoder(
            spec.acoustic_tokenizer_encoder, model.acoustic_tokenizer_encoder
        )

        print("Setting semantic tokenizer encoder weights ...")
        self._set_acoustic_encoder(
            spec.semantic_tokenizer_encoder, model.semantic_tokenizer_encoder
        )

        print("Setting multi-modal projector weights ...")
        self._set_multi_modal_projector(
            spec.multi_modal_projector, model.multi_modal_projector
        )

        print("Setting Qwen2 decoder weights ...")
        self._set_qwen2_decoder(spec.decoder, model.language_model)

        return spec

    # ------------------------------------------------------------------
    # Acoustic / Semantic encoder の重みマッピング
    # ------------------------------------------------------------------

    def _set_acoustic_encoder(self, spec, encoder_module):
        """VAE エンコーダ（ConvNeXt 系 CNN）の重みを spec にセットする。

        encoder_module の構造 (Transformers v5.4.0):
          .stem
            .conv   CausalConv1d → .conv → nn.Conv1d
            .stage  list of VibeVoiceAcousticTokenizerConvNext1dLayer
          .conv_layers  list of VibeVoiceAcousticTokenizerDownsamplingLayer
            .conv   CausalConv1d → .conv → nn.Conv1d (strided)
            .stage  list of ConvNext1dLayer
          .head  CausalConv1d → .conv → nn.Conv1d
        """
        # stem: 最初の CausalConv1d
        self._set_causal_conv1d(spec.stem.conv, encoder_module.stem.conv)

        # stem: ConvNeXt ブロック列
        for i, block in enumerate(encoder_module.stem.stage):
            self._set_convnext_block(spec.stem.stage[i], block)

        # ダウンサンプリング層（6 段）＋ 各段の ConvNeXt ブロック列
        for i, layer in enumerate(encoder_module.conv_layers):
            self._set_causal_conv1d(spec.conv_layers[i].conv, layer.conv)
            for j, block in enumerate(layer.stage):
                self._set_convnext_block(spec.conv_layers[i].stage[j], block)

        # head: 最終射影 CausalConv1d
        self._set_causal_conv1d(spec.head, encoder_module.head)

    def _set_convnext_block(self, spec, block):
        """VibeVoiceAcousticTokenizerConvNext1dLayer の重みをセット。

        Sub-block 1 (mixer path):
          block.norm    → spec.norm     (RMSNorm, weight のみ)
          block.mixer   → spec.mixer    (CausalConv1d, depth-wise)
          block.gamma   → spec.gamma    (LayerScale nn.Parameter [C])

        Sub-block 2 (FFN path):
          block.ffn_norm      → spec.ffn_norm
          block.ffn.linear1   → spec.ffn_linear1
          block.ffn.linear2   → spec.ffn_linear2
          block.ffn_gamma     → spec.ffn_gamma  (nn.Parameter [C])
        """
        # Sub-block 1
        self._set_layer_norm(spec.norm, block.norm)
        self._set_causal_conv1d(spec.mixer, block.mixer)
        spec.gamma = block.gamma.detach()

        # Sub-block 2
        self._set_layer_norm(spec.ffn_norm, block.ffn_norm)
        self._set_linear(spec.ffn_linear1, block.ffn.linear1)
        self._set_linear(spec.ffn_linear2, block.ffn.linear2)
        spec.ffn_gamma = block.ffn_gamma.detach()

    # ------------------------------------------------------------------
    # Multi-modal projector の重みマッピング
    # ------------------------------------------------------------------

    def _set_multi_modal_projector(self, spec, projector):
        """VibeVoiceAsrMultiModalProjector の重みをセット。

        projector:
          .acoustic_linear_1  nn.Linear
          .acoustic_norm      RMSNorm
          .acoustic_linear_2  nn.Linear
          .semantic_linear_1  nn.Linear
          .semantic_norm      RMSNorm
          .semantic_linear_2  nn.Linear
        """
        self._set_linear(spec.acoustic_linear_1, projector.acoustic_linear_1)
        self._set_layer_norm(spec.acoustic_norm, projector.acoustic_norm)
        self._set_linear(spec.acoustic_linear_2, projector.acoustic_linear_2)
        self._set_linear(spec.semantic_linear_1, projector.semantic_linear_1)
        self._set_layer_norm(spec.semantic_norm, projector.semantic_norm)
        self._set_linear(spec.semantic_linear_2, projector.semantic_linear_2)

    # ------------------------------------------------------------------
    # Qwen2 デコーダの重みマッピング（transformers.py の Qwen2Loader から流用）
    # ------------------------------------------------------------------

    def _set_qwen2_decoder(self, spec, lm_model):
        """Qwen2ForCausalLM の重みを TransformerDecoderSpec にセットする。"""
        decoder = lm_model.model

        spec.scale_embeddings = False
        self._set_embeddings(spec.embeddings, decoder.embed_tokens)
        self._set_layer_norm(spec.layer_norm, decoder.norm)

        for layer_spec, layer in zip(spec.layer, decoder.layers):
            self._set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self._set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            # QKV を結合（CTranslate2 は QKV fused linear を期待）
            from ctranslate2.converters import utils as ct2_utils

            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self._set_linear(split_layers[0], layer.self_attn.q_proj)
            self._set_linear(split_layers[1], layer.self_attn.k_proj)
            self._set_linear(split_layers[2], layer.self_attn.v_proj)
            ct2_utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)

            self._set_linear(
                layer_spec.self_attention.linear[1], layer.self_attn.o_proj
            )
            self._set_linear(layer_spec.ffn.linear_0, layer.mlp.gate_proj)
            self._set_linear(layer_spec.ffn.linear_0_noact, layer.mlp.up_proj)
            self._set_linear(layer_spec.ffn.linear_1, layer.mlp.down_proj)

            # メモリ節約: 処理済み層を削除
            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()

        # lm_head
        self._set_linear(spec.projection, lm_model.lm_head)

    # ------------------------------------------------------------------
    # config / vocabulary / files
    # ------------------------------------------------------------------

    def _set_config(self, spec, model):
        cfg = model.config
        enc_cfg = cfg.acoustic_tokenizer_encoder_config
        sem_cfg = cfg.semantic_tokenizer_encoder_config
        lm_cfg = cfg.text_config

        spec.config.__dict__.update(
            VibeVoiceAsrConfig(
                audio_token_id=cfg.audio_token_id,
                audio_bos_token_id=cfg.audio_bos_token_id,
                audio_eos_token_id=cfg.audio_eos_token_id,
                acoustic_tokenizer_chunk_size=cfg.acoustic_tokenizer_chunk_size,
                acoustic_hidden_size=enc_cfg.hidden_size,
                semantic_hidden_size=sem_cfg.hidden_size,
                acoustic_downsampling_ratios=list(enc_cfg.downsampling_ratios),
                acoustic_depths=list(enc_cfg.depths),
                acoustic_num_filters=enc_cfg.num_filters,
                acoustic_kernel_size=enc_cfg.kernel_size,
                acoustic_vae_std=float(enc_cfg.vae_std),
                decoder_hidden_size=lm_cfg.hidden_size,
                decoder_num_layers=lm_cfg.num_hidden_layers,
                decoder_num_heads=lm_cfg.num_attention_heads,
                decoder_num_kv_heads=lm_cfg.num_key_value_heads,
                decoder_intermediate_size=lm_cfg.intermediate_size,
            ).to_dict()
        )

    def _set_vocabulary(self, spec, tokenizer, vocab_size=None):
        vocab = [
            token
            for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
        ]
        if vocab_size is not None and len(vocab) < vocab_size:
            for i in range(vocab_size - len(vocab)):
                vocab.append("<extra_id_%d>" % i)
        spec.register_vocabulary(vocab)

    def _register_copy_files(self, spec, processor):
        """プロセッサ関連ファイルを変換先ディレクトリにコピー登録する。"""
        # Try to find files in the original model directory first, then fall back
        # to saving the processor to a persistent temp dir.
        # NOTE: do NOT use a context-manager tempdir here — register_file() only
        # stores the path, and the actual copy happens later in model_spec.save().
        import tempfile

        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        processor.save_pretrained(tmpdir)

        for filename in self._copy_files:
            # Prefer the file from the original model path (always up-to-date).
            src = os.path.join(self._model_path, filename)
            if not os.path.isfile(src):
                src = os.path.join(tmpdir, filename)
            if os.path.isfile(src):
                spec.register_file(src, filename=filename)
            else:
                print(f"[警告] copy_files: {filename} が見つかりません（スキップ）")

        # Keep tmpdir_obj alive so files remain accessible until save() completes.
        self._tmpdir_obj = tmpdir_obj

    # ------------------------------------------------------------------
    # 低レベルヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _set_causal_conv1d(spec, causal_conv_module):
        """CausalConv1d ラッパーの内部 nn.Conv1d (.conv) から重みをセット。"""
        inner = causal_conv_module.conv  # nn.Conv1d
        spec.weight = inner.weight.detach()
        if inner.bias is not None:
            spec.bias = inner.bias.detach()

    @staticmethod
    def _set_layer_norm(spec, module):
        spec.gamma = module.weight.detach()
        if hasattr(module, "bias") and module.bias is not None:
            spec.beta = module.bias.detach()

    @staticmethod
    def _set_linear(spec, module):
        spec.weight = module.weight.detach()
        if hasattr(module, "bias") and module.bias is not None:
            spec.bias = module.bias.detach()

    @staticmethod
    def _set_embeddings(spec, module):
        spec.weight = module.weight.detach()

    # ------------------------------------------------------------------
    # CLI エントリポイント
    # ------------------------------------------------------------------

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(
            description="Convert microsoft/VibeVoice-ASR-HF to CTranslate2 format."
        )
        parser.add_argument(
            "--model",
            required=True,
            help="HuggingFace model ID or local path to VibeVoice-ASR-HF.",
        )
        parser.add_argument(
            "--copy_files",
            nargs="*",
            default=["tokenizer.json", "preprocessor_config.json",
                     "processor_config.json", "tokenizer_config.json",
                     "special_tokens_map.json", "chat_template.jinja"],
            help="Additional files from the processor to copy to the output directory.",
        )
        parser.add_argument(
            "--low_cpu_mem_usage",
            action="store_true",
            default=True,
            help="Load the model with low CPU memory usage (default: True).",
        )
        parser.add_argument(
            "--no_low_cpu_mem_usage",
            action="store_false",
            dest="low_cpu_mem_usage",
            help="Disable low CPU memory usage mode.",
        )
        parser = Converter.declare_arguments(parser)
        args = parser.parse_args()

        converter = cls(
            model_name_or_path=args.model,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            copy_files=args.copy_files,
        )
        output_dir = converter.convert_from_args(args)
        print(f"Model converted to: {output_dir}")


def main():
    VibeVoiceAsrConverter.main()
