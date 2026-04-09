# Gemma 4 Multimodal

CTranslate2 supports **Gemma 4** (`Gemma4ForConditionalGeneration`) — Google's multimodal model
that pairs a SigLIP-based vision encoder with a Gemma 4 text decoder for vision-conditioned
text generation.

```{note}
This guide covers a model specific to the
[PyRin-c/CTranslate2](https://github.com/PyRin-c/CTranslate2) fork
(`feature/add_support_parakeet-ctc`).
The `Gemma4` class is not present in upstream OpenNMT/CTranslate2.
```

## Architecture overview

| Component | Description |
|---|---|
| `vision_encoder` | SigLIP transformer — patch embedder (pixel → patch tokens with 2-D learned positional embeddings), N layers of pre/post-norm self-attention (GeGLU FFN, per-head q/k/v norm, 2-D RoPE), spatial 2-D average pool |
| `multimodal_embedder` | Scaleless RMSNorm + linear projection from vision hidden size → text hidden size |
| `decoder` | Gemma 4 text decoder — GQA, QK-norm, pre-post LayerNorm, proportional RoPE, optional MoE, optional sliding-window attention |

The model works in two phases at inference time:

1. **Vision encoding**: raw patches → pooled vision embeddings via `encode_vision()`
2. **Multimodal generation**: text tokens (with image placeholder tokens replaced by vision embeddings) → autoregressive text via `generate()`

## Requirements

```bash
pip install ctranslate2 torch transformers pillow
```

GPU inference requires a CUDA-enabled CTranslate2 build. On Windows, add the cuDNN DLL
directory before importing:

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64")
import ctranslate2
```

## Converting a HuggingFace checkpoint

Use `ct2-transformers-converter`. The converter automatically dispatches to
`Gemma4MultimodalLoader` when it detects a `Gemma4Config` model.

**bfloat16 (recommended for modern GPUs):**

```bash
ct2-transformers-converter \
    --model google/gemma-4-4b-it \
    --output_dir ./gemma4-ct2 \
    --quantization bfloat16 \
    --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json
```

**float32 (no quantization):**

```bash
ct2-transformers-converter \
    --model google/gemma-4-4b-it \
    --output_dir ./gemma4-ct2-f32
```

The output directory contains `config.json`, `model.bin`, and vocabulary files.
`config.json` includes `image_token_id` (the placeholder token used in the prompt).

## Inference

### Image preprocessing

Before calling CTranslate2, preprocess images with HuggingFace's `AutoProcessor`.
The preprocessor returns `pixel_values` and `pixel_position_ids` that are passed directly to
`encode_vision()`.

```python
from PIL import Image
from transformers import AutoProcessor

MODEL_ID = "google/gemma-4-4b-it"   # or path to local HF checkpoint
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

### End-to-end example

```python
import json
import numpy as np
from PIL import Image
from transformers import AutoProcessor
import ctranslate2
from ctranslate2.models import Gemma4

MODEL_DIR = "./gemma4-ct2"
DEVICE    = "cuda"   # or "cpu"
HF_ID     = "google/gemma-4-4b-it"

# Load CTranslate2 model
model = Gemma4(MODEL_DIR, device=DEVICE)

# Load config (vocabulary, special token IDs)
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)
IMAGE_TOKEN_ID = cfg["image_token_id"]      # e.g. 262144

# Load HuggingFace processor for tokenisation and image preprocessing
processor = AutoProcessor.from_pretrained(HF_ID)

# -----------------------------------------------------------------
# 1. Preprocess image
# -----------------------------------------------------------------
image = Image.open("photo.jpg").convert("RGB")

# Build a minimal chat message with a single image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": "Describe this image in detail."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="np",
)

# pixel_values:      (1, num_patches, patch_dim) float32
# pixel_position_ids: (1, num_patches, 2)        int32
pixel_values     = inputs["pixel_values"].astype(np.float32)   # (1, N, D)
pixel_pos_ids    = inputs["pixel_position_ids"].astype(np.int32)
input_ids        = inputs["input_ids"].astype(np.int32)         # (1, seq_len)

# -----------------------------------------------------------------
# 2. Encode vision
# -----------------------------------------------------------------
pv_sv   = ctranslate2.StorageView.from_array(pixel_values)
pp_sv   = ctranslate2.StorageView.from_array(pixel_pos_ids)
vision_embeds = model.encode_vision(pv_sv, pp_sv)   # (1, num_patches, text_hidden)

# -----------------------------------------------------------------
# 3. Generate
# -----------------------------------------------------------------
ids_sv = ctranslate2.StorageView.from_array(input_ids)

results = model.generate(
    ids_sv,
    vision_embeds,
    max_new_tokens=256,
    repetition_penalty=1.1,
)

# -----------------------------------------------------------------
# 4. Decode output
# -----------------------------------------------------------------
with open(f"{MODEL_DIR}/vocabulary.json", encoding="utf-8") as f:
    vocab = json.load(f)

# Standard SentencePiece byte-fallback decode
def decode_ids(ids):
    tokens = [vocab[i] for i in ids]
    text = "".join(tokens)
    # Remove sentencepiece leading space marker
    return text.replace("▁", " ").strip()

print(decode_ids(results[0].sequences_ids[0]))
```

### Multiple images

Pass one image's `pixel_values` / `pixel_position_ids` per call; batching across multiple
images in a single call is not yet supported.

### Greedy vs. beam search

```python
# Greedy (default, beam_size=1)
results = model.generate(ids_sv, vision_embeds, max_new_tokens=256)

# Beam search (4 beams, return top-2 hypotheses)
results = model.generate(
    ids_sv,
    vision_embeds,
    max_new_tokens=256,
    beam_size=4,
    num_hypotheses=2,
    return_scores=True,
)
print(results[0].scores)          # log-prob per hypothesis
print(results[0].sequences_ids)   # list of token ID lists, one per hypothesis
```

### Sampling

```python
results = model.generate(
    ids_sv,
    vision_embeds,
    max_new_tokens=256,
    sampling_topk=50,
    sampling_temperature=0.8,
)
```

## Quantization

```python
# bfloat16 weights (recommended for modern GPUs)
model_bf16 = Gemma4(MODEL_DIR, device="cuda", compute_type="bfloat16")

# float16 weights
model_fp16 = Gemma4(MODEL_DIR, device="cuda", compute_type="float16")

# int8 weights (smallest memory footprint)
model_int8 = Gemma4(MODEL_DIR, device="cpu", compute_type="int8")
```

Supported `compute_type` values: `float32`, `float16`, `bfloat16`, `int8`, `int8_float16`,
`int8_float32`.

```{note}
Quantization applies to linear layers in the text decoder and the multimodal embedder.
The vision encoder (SigLIP) always runs in the model's floating-point dtype.
```

## Python API reference

```python
from ctranslate2.models import Gemma4, Gemma4GenerationResult

model = Gemma4(
    model_path,              # path to the converted model directory
    device="cpu",            # "cpu" or "cuda"
    compute_type="default",
    inter_threads=1,
    intra_threads=0,
)

# Read the image placeholder token ID stored in config.json
image_token_id: int = model.image_token_id   # e.g. 262144

# Encode raw patches to vision embeddings
# pixel_values:       StorageView (batch, num_patches, patch_dim) float32
# pixel_position_ids: StorageView (batch, num_patches, 2)         int32
vision_embeds = model.encode_vision(pixel_values, pixel_position_ids)
# Returns: StorageView (batch, num_patches, text_hidden_size)

# Generate text conditioned on vision embeddings
# input_ids:    StorageView (batch, seq_len) int32
#               — full prompt with image_token_id placeholders at image positions
# vision_embeds: output of encode_vision()
results: list[Gemma4GenerationResult] = model.generate(
    input_ids,
    vision_embeds,
    max_new_tokens=256,         # default 256
    beam_size=1,                # default 1 (greedy); >1 enables beam search
    patience=1.0,
    length_penalty=1.0,
    repetition_penalty=1.0,     # recommend 1.1 to suppress repetition
    sampling_topk=1,            # default 1 (greedy); >1 enables top-k sampling
    sampling_temperature=1.0,
    num_hypotheses=1,
    return_scores=False,
    image_token_id=0,           # 0 → use value from config.json automatically
)

# Gemma4GenerationResult fields:
#   sequences        list[list[str]]   — token strings
#   sequences_ids    list[list[int]]   — token IDs
#   scores           list[float]       — log-probs (empty if return_scores=False)
```

## Conversion notes

- The vision encoder uses **2-D RoPE**: separate x-axis and y-axis position IDs are gathered
  from precomputed cosine/sine tables and concatenated to form the full rotary encoding.
- The multimodal embedder applies a **scaleless RMSNorm** (gamma stored as ones) before the
  linear projection.
- The text decoder uses **`scale_embeddings=False`** in the spec; the embedding scale
  (`sqrt(text_hidden_size)`) is applied inside `generate()` after replacing image placeholders.
- `image_token_id` is written to `config.json` by the converter and is automatically read
  by `model.image_token_id` and used as the default in `generate()`.
