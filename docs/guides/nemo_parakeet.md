# NeMo Parakeet

CTranslate2 supports NVIDIA's **Parakeet** family of speech recognition models converted from [NeMo](https://github.com/NVIDIA/NeMo) checkpoints.

```{note}
This guide covers models specific to the [PyRin-c/CTranslate2](https://github.com/PyRin-c/CTranslate2) fork (`feature/add_support_parakeet-ctc`).
The `Parakeet` class and the `ct2-nemo-parakeet-converter` tool are not present in upstream OpenNMT/CTranslate2.
```

## Supported models

| Model family | Decoder | Example checkpoint |
|---|---|---|
| `parakeet-tdt_ctc` | CTC and TDT | `nvidia/parakeet-tdt_ctc-0.6b` |
| `parakeet-tdt` | TDT only | `nvidia/parakeet-tdt-0.6b-v3` |

Both English and Japanese variants are supported. The model architecture is a 24-layer **Conformer** encoder followed by either a CTC head or a TDT (Token-and-Duration Transducer) prediction network.

## Requirements

```bash
pip install ctranslate2 torch torchaudio librosa soundfile sentencepiece
```

GPU inference requires a CUDA-enabled build of CTranslate2. On Windows, the cuDNN DLL directory must be reachable before importing `ctranslate2`:

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64")
import ctranslate2
```

## Converting a NeMo checkpoint

Use the `ct2-nemo-parakeet-converter` command-line tool to convert a `.nemo` file to the CTranslate2 model directory format.

**Convert with float32 weights (no quantization):**

```bash
ct2-nemo-parakeet-converter \
    --model /path/to/parakeet-tdt_ctc-0.6b.nemo \
    --output_dir ./parakeet_ct2
```

**Convert with float16 quantization (recommended for GPU):**

```bash
ct2-nemo-parakeet-converter \
    --model /path/to/parakeet-tdt_ctc-0.6b.nemo \
    --output_dir ./parakeet_ct2_fp16 \
    --quantization float16
```

The output directory will contain `config.json`, `model.bin`, and the SentencePiece tokenizer file (`.model`).

```{note}
`torch` is required for conversion. `nemo_toolkit[asr]` is only required when passing a NeMo model ID (e.g. `nvidia/parakeet-tdt_ctc-0.6b`) instead of a local `.nemo` file path.
```

## Preparing the input: log-mel spectrogram

CTranslate2's `Parakeet` model expects a **log-mel spectrogram** as input — it does not perform audio preprocessing internally. The spectrogram must be computed to match NeMo's `AudioToMelSpectrogramPreprocessor` in eval mode.

```python
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf

SR      = 16000
N_FFT   = 512
HOP_LEN = 160    # 0.01 s
WIN_LEN = 400    # 0.025 s
N_MELS  = 80
PREEMPH = 0.97

def compute_mel(audio_path: str) -> np.ndarray:
    """Return log-mel spectrogram of shape [80, T] compatible with NeMo eval mode."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio)
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)

    # Pre-emphasis
    waveform = torch.cat([waveform[:1], waveform[1:] - PREEMPH * waveform[:-1]])

    # STFT power spectrum
    window = torch.hann_window(WIN_LEN)
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
                      window=window, center=True, pad_mode="constant", return_complex=True)
    power = stft.abs().pow(2)

    # Mel filterbank (norm='slaney')
    mel_fb = torch.tensor(
        librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, norm="slaney"),
        dtype=torch.float32)
    mel = torch.matmul(mel_fb, power)

    # Log compression and per-feature normalization
    log_mel = torch.log(mel + 2.0 ** -24)
    mean = log_mel.mean(dim=1, keepdim=True)
    std  = log_mel.std(dim=1, keepdim=True)
    log_mel = (log_mel - mean) / (std + 1e-5)

    return log_mel.numpy()  # [80, T]
```

## CTC inference

CTC decoding collapses repeated tokens and removes blank symbols, producing a flat sequence of token IDs.

```python
import os
import numpy as np
import sentencepiece as spm
import ctranslate2
from ctranslate2.models import Parakeet

MODEL_DIR = "./parakeet_ct2"
AUDIO_PATH = "audio.wav"
DEVICE = "cuda"  # or "cpu"

# Load tokenizer and model
sp = spm.SentencePieceProcessor()
sp.Load(os.path.join(MODEL_DIR, "tokenizer.model"))

model = Parakeet(MODEL_DIR, device=DEVICE)

# Prepare input
mel_np = compute_mel(AUDIO_PATH)[np.newaxis, :, :]  # [1, 80, T]
T = mel_np.shape[2]
lens_np = np.array([T], dtype=np.int32)

mel_sv  = ctranslate2.StorageView.from_array(mel_np)
lens_sv = ctranslate2.StorageView.from_array(lens_np)

# CTC transcription
result = model.transcribe(mel_sv, lens_sv, use_tdt=False)
ids = result.ids[0]
text = sp.DecodeIds([i for i in ids if i != sp.GetPieceSize()])
print(text)
```

## TDT inference with timestamps

TDT (Token-and-Duration Transducer) decoding additionally provides the encoder-frame index at which each token was emitted, enabling word-level timestamp estimation.

```python
# TDT transcription
result = model.transcribe(mel_sv, lens_sv, use_tdt=True)
ids = result.ids[0]
text = sp.DecodeIds([i for i in ids if i != sp.GetPieceSize()])
print(text)

# Token-level timestamps
# The Conformer encoder subsamples mel by a factor of 8 (3 stride-2 conv layers).
# time_s = encoder_frame * 8 * HOP_LEN / SR
SUBSAMPLE = 8
FRAME_SHIFT_S = HOP_LEN / SR  # 0.01 s

for token_id, enc_frame in zip(ids, result.token_start_frames[0]):
    token_text = sp.IdToPiece(token_id).replace("\u2581", " ").strip()
    t_sec = enc_frame * SUBSAMPLE * FRAME_SHIFT_S
    print(f"  {t_sec:6.3f}s  {token_text!r}")
```

`token_start_frames` is empty when `use_tdt=False`.

## Batch inference

Pass multiple spectrograms as a batch. If the lengths differ, pad the shorter ones along the time axis and provide the actual lengths in `lens_sv`.

```python
# Two clips of the same length (no padding needed)
mel_batch  = np.concatenate([mel_np, mel_np], axis=0)  # [2, 80, T]
lens_batch = np.array([T, T], dtype=np.int32)

mel_batch_sv  = ctranslate2.StorageView.from_array(mel_batch)
lens_batch_sv = ctranslate2.StorageView.from_array(lens_batch)

result = model.transcribe(mel_batch_sv, lens_batch_sv, use_tdt=False)
for i, ids in enumerate(result.ids):
    print(f"[{i}] {sp.DecodeIds(ids)}")
```

## Encoder-only output

Use `encode` when you only need the Conformer encoder's output — for example, to feed into a downstream model.

```python
future = model.encode(mel_sv, lens_sv)
enc_out = np.array(future.get())  # [batch, T', 1024]
print(enc_out.shape)
```

## Quantization

The model can be loaded with reduced precision to trade accuracy for speed and memory.

```python
# Load with float16 weights (GPU recommended)
model_fp16 = Parakeet(MODEL_DIR, device="cuda", compute_type="float16")

# Load with int8 quantization (CPU or GPU)
model_int8 = Parakeet(MODEL_DIR, device="cpu", compute_type="int8")
```

Supported `compute_type` values: `float32`, `float16`, `bfloat16`, `int8`, `int8_float16`, `int8_float32`, `int16`.

```{note}
Quantization is applied to linear layer weights. The Conv2D and RelShift operations inside the Conformer always run in floating point.
```

## Python API reference

```python
from ctranslate2.models import Parakeet

model = Parakeet(
    model_path,          # path to the converted model directory
    device="cpu",        # "cpu" or "cuda"
    compute_type="default",
    inter_threads=1,
    intra_threads=0,
)

# Full transcription
result = model.transcribe(
    mel,                 # StorageView [batch, 80, T]
    lengths,             # StorageView [batch]  (actual T per item)
    use_tdt=False,       # True → TDT decoder, False → CTC decoder
    max_tdt_steps=0,     # TDT safety limit (0 = encoder_length * 2)
).get()

# result.ids                  list[list[int]]   token ids per batch item
# result.token_start_frames   list[list[int]]   encoder frame indices (TDT only)

# Encoder output only
enc_out = model.encode(mel, lengths).get()  # StorageView [batch, T', 1024]
```
