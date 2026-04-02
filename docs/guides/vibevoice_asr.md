# VibeVoice-ASR

CTranslate2 supports **VibeVoice-ASR** — Microsoft's speech recognition model that combines a
CausalConv1d / ConvNeXt acoustic encoder with a Qwen2 28-layer decoder-only LM.

```{note}
This guide covers a model specific to the
[PyRin-c/CTranslate2](https://github.com/PyRin-c/CTranslate2) fork
(`feature/support_vibevoice`).
The `VibeVoiceAsr` class and the `ct2-vibevoice-asr-converter` tool are not present in
upstream OpenNMT/CTranslate2.
```

## Architecture overview

| Component | Description |
|---|---|
| `acoustic_tokenizer_encoder` | CausalConv1d stem + 6 strided downsampling layers of ConvNeXt blocks → hidden_size=64 |
| `semantic_tokenizer_encoder` | Identical structure, hidden_size=128 |
| `multi_modal_projector` | Two linear+RMSNorm+linear pathways that map acoustic and semantic features to the LM hidden size (3584) |
| `decoder` | 28-layer Qwen2 decoder-only LM, GQA (28 heads / 4 KV heads) |

The model takes a raw waveform at 24 kHz, encodes it into audio embeddings, injects those
embeddings into the text token sequence at positions marked with a special `audio_token_id`,
and auto-regressively generates the transcription.

## Requirements

```bash
pip install ctranslate2 torch transformers
```

GPU inference requires a CUDA-enabled CTranslate2 build.  On Windows, add the cuDNN DLL
directory before importing:

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64")
import ctranslate2
```

## Converting a HuggingFace checkpoint

Use `ct2-vibevoice-asr-converter` to convert the HuggingFace model to the CTranslate2
directory format.

**bfloat16 (recommended):**

```bash
ct2-vibevoice-asr-converter \
    --model microsoft/VibeVoice-ASR-HF \
    --output_dir ./vibevoice-asr-ct2 \
    --quantization bfloat16 \
    --copy_files tokenizer.json preprocessor_config.json
```

**float32 (no quantization):**

```bash
ct2-vibevoice-asr-converter \
    --model microsoft/VibeVoice-ASR-HF \
    --output_dir ./vibevoice-asr-ct2-f32
```

The output directory will contain `config.json`, `model.bin`, and the vocabulary file.

## Inference

### End-to-end example

The model follows the Qwen2 chat format.  The input sequence must include the
full system/user prompt — **not** just the audio placeholder tokens.

```python
import json
import numpy as np
import ctranslate2
from ctranslate2.models import VibeVoiceAsr
from tokenizers import Tokenizer          # pip install tokenizers

MODEL_DIR = "./vibevoice-asr-ct2"
DEVICE    = "cuda"   # or "cpu"

# Load model
model = VibeVoiceAsr(MODEL_DIR, device=DEVICE)

# Load config (special token IDs)
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)
AUDIO_TOKEN_ID = cfg["audio_token_id"]       # 151648  audio frame placeholder
AUDIO_BOS_ID   = cfg["audio_bos_token_id"]   # 151646  <|object_ref_start|>
AUDIO_EOS_ID   = cfg["audio_eos_token_id"]   # 151647  <|object_ref_end|>
IM_START = 151644   # <|im_start|>
IM_END   = 151645   # <|im_end|>
NL       = 198      # newline token

# Tokenizer for encoding plain text
tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")

# Prepare waveform: (1, num_samples) float32 at 24 kHz
duration_sec = 5.0
waveform = np.zeros((1, int(24000 * duration_sec)), dtype=np.float32)
waveform_sv = ctranslate2.StorageView.from_array(waveform)

# 1. Encode waveform → audio features  (1, T, 3584)
audio_features = model.encode(waveform_sv)
T = np.array(audio_features).shape[1]   # number of audio frames

# 2. Build input_ids following the chat template
#    <|im_start|>system\n{system_prompt}<|im_end|>\n
#    <|im_start|>user\n<audio_bos><audio_token×T><audio_eos>\n{instruction}<|im_end|>\n
#    <|im_start|>assistant\n                   ← model generates from here
system_ids = tok.encode(
    "You are a helpful assistant that transcribes audio input into text output in JSON format."
).ids
instr_ids = tok.encode(
    f"This is a {duration_sec:.1f} seconds audio, "
    "please transcribe it with these keys: Start time, End time, Speaker ID, Content"
).ids

input_ids = []
# system turn
input_ids += [IM_START, 8948, NL]   # 8948 = "system"
input_ids += system_ids
input_ids += [IM_END, NL]
# user turn
input_ids += [IM_START, 872, NL]    # 872 = "user"
input_ids += [AUDIO_BOS_ID]
input_ids += [AUDIO_TOKEN_ID] * T   # one placeholder per audio frame
input_ids += [AUDIO_EOS_ID]
input_ids += [NL]
input_ids += instr_ids
input_ids += [IM_END, NL]
# start of assistant turn (model continues from here)
input_ids += [IM_START, 77091, NL]  # 77091 = "assistant"

input_ids_sv = ctranslate2.StorageView.from_array(
    np.array([input_ids], dtype=np.int32)
)

# 3. Generate transcription
results = model.generate(input_ids_sv, audio_features,
                         max_new_tokens=512, repetition_penalty=1.1)

# 4. Decode output token IDs to text (GPT-2 byte-level BPE)
with open(f"{MODEL_DIR}/vocabulary.json", encoding="utf-8") as f:
    vocab = json.load(f)

# Build byte_decoder (standard GPT-2 approach)
_bs = (list(range(ord("!"), ord("~") + 1))
       + list(range(ord("\xa1"), ord("\xac") + 1))
       + list(range(ord("\xae"), ord("\xff") + 1)))
_cs, _n = _bs[:], 0
for b in range(256):
    if b not in _bs:
        _bs.append(b); _cs.append(256 + _n); _n += 1
byte_decoder = {chr(c): b for b, c in zip(_bs, _cs)}

def decode_ids(ids):
    raw = []
    for i in ids:
        for ch in vocab[i]:
            raw.append(byte_decoder[ch] if ch in byte_decoder else ord(ch))
    return bytes(raw).decode("utf-8", errors="replace")

print(decode_ids(results[0].sequences_ids[0]))
```

### Chunked encoding for long audio

For audio longer than ~60 seconds, pass `chunk_size` to `encode()` to limit peak memory:

```python
# Process a 2-minute waveform in 60-second chunks (default chunk_size = 1 440 000 samples)
long_waveform = np.zeros((1, 24000 * 120), dtype=np.float32)  # 2 minutes
long_sv = ctranslate2.StorageView.from_array(long_waveform)

audio_features = model.encode(long_sv, chunk_size=1_440_000)
T = np.array(audio_features).shape[1]
```

The default `chunk_size` is 1 440 000 samples (= 60 s at 24 kHz).

## Quantization

```python
# bfloat16 weights (recommended for modern GPUs)
model_bf16 = VibeVoiceAsr(MODEL_DIR, device="cuda", compute_type="bfloat16")

# float16 weights
model_fp16 = VibeVoiceAsr(MODEL_DIR, device="cuda", compute_type="float16")

# int8 weights (smallest memory footprint)
model_int8 = VibeVoiceAsr(MODEL_DIR, device="cpu", compute_type="int8")
```

Supported `compute_type` values: `float32`, `float16`, `bfloat16`, `int8`,
`int8_float16`, `int8_float32`.

```{note}
Quantization applies to linear layers in the Qwen2 decoder and the multi-modal projector.
The CausalConv1d and ConvNeXt acoustic encoder layers always run in the model's floating-point
dtype and are not quantized to int8/int16.
```

## Python API reference

```python
from ctranslate2.models import VibeVoiceAsr, VibeVoiceAsrResult

model = VibeVoiceAsr(
    model_path,              # path to the converted model directory
    device="cpu",            # "cpu" or "cuda"
    compute_type="default",
    inter_threads=1,
    intra_threads=0,
)

# Encode waveform → audio feature embeddings
# waveform_sv: StorageView of shape (batch, num_samples) float32, 24 kHz
audio_features = model.encode(waveform_sv)          # StorageView (batch, T, 3584)
audio_features = model.encode(waveform_sv,
                               chunk_size=1_440_000) # with explicit chunk size

# Generate transcription tokens
# input_ids_sv:   StorageView (batch, seq_len) int32
#                 — full chat-format prompt with audio_token_id placeholders (see example above)
# audio_features: output of encode()
results: list[VibeVoiceAsrResult] = model.generate(
    input_ids_sv, audio_features,
    max_new_tokens=512,         # default 8192
    beam_size=1,                # default 1 (greedy)
    patience=1.0,
    length_penalty=1.0,
    repetition_penalty=1.1,     # recommended: 1.1 to suppress looping
    sampling_topk=1,
    sampling_temperature=1.0,
    num_hypotheses=1,
    return_scores=False,
    audio_token_id=151648,      # default, matches config.json
)

# VibeVoiceAsrResult fields:
#   sequences        list[list[str]]   — raw BPE token strings (need byte-level decode)
#   sequences_ids    list[list[int]]   — token IDs
#   scores           list[float]       — log-probs (empty if return_scores=False)
```
