# TransNetV2 (PyTorch) — Scene / Shot Boundary Detection with NVDEC (optional) + PyAV fallback

This repository is a fork of [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) with a **PyTorch inference pipeline** and a clean, OOP-based API.

It supports two decoding backends:

- **NVIDIA NVDEC (GPU decode)** via **PyNvVideoCodec** *(optional)* — fastest path when the codec is supported by your GPU.
- **PyAV (FFmpeg, CPU decode)** — always available fallback (and the default when you run on CPU).

> When you run on **CUDA**, the library tries NVDEC first and **automatically falls back** to PyAV if NVDEC can’t decode the input (for example: unsupported codec/profile/chroma on this GPU).

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Configuration](#configuration)
- [Logging](#logging)
- [License](#license)

---

## Features

- **PyTorch model**: loads TransNetV2 weights with PyTorch and runs inference on CPU or CUDA.
- **Automatic backend selection**:
  - `device="cpu"` → **PyAV**
  - `device="cuda"` → try **NVDEC (PyNvVideoCodec)**, otherwise **PyAV**
- **Progress bars**: optional `tqdm` progress bars during decoding / window processing.
- **Clean API**: a single entry point class: `SceneDetector`.

---

## Installation

### 1) Install Python deps

```bash
pip install -r requirements.txt
```

### 2) Install PyTorch

Use the official selector to pick the correct CUDA / CPU build:
- https://pytorch.org/get-started/locally/

### 3) Install PyAV (CPU backend)

PyAV provides binary wheels on PyPI for Windows / Linux / macOS:

```bash
pip install av
```

Docs:
- https://pyav.org/docs/develop/overview/installation.html

### 4) (Optional) Install PyNvVideoCodec (NVDEC backend)

If you want **GPU-accelerated decode**, install NVIDIA **PyNvVideoCodec** (requires NVIDIA driver + compatible GPU):

- https://developer.nvidia.com/pynvvideocodec
- API Programming Guide: https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html

> NVDEC codec support depends on the GPU and the codec/profile of the input video. If NVDEC can’t decode your video, the library will fall back to PyAV automatically.

---

## Usage

### Basic (auto device selection)

```python
from transnetv2pt import SceneDetector

detector = SceneDetector()  # CUDA if available else CPU
scenes = detector.predict("path/to/video.mp4", show_progressbar=True)

print(scenes)  # [[start_frame, end_frame], ...]
```

### Force CUDA (NVDEC first, fallback to PyAV if unsupported)

```python
from transnetv2pt import SceneDetector
import torch

detector = SceneDetector(torch.device("cuda"))
scenes = detector.predict("path/to/video.mp4", show_progressbar=True)
```

### Force CPU (PyAV only)

```python
from transnetv2pt import SceneDetector
import torch

detector = SceneDetector(torch.device("cpu"))
scenes = detector.predict("path/to/video.mp4", show_progressbar=True)
```

---

## Examples

### Extract keyframes at scene starts (OpenCV)

```python
from pathlib import Path
import cv2
import torch
from transnetv2pt import SceneDetector

video_path = Path("video.mkv")

detector = SceneDetector(torch.device("cuda"))  # or "cpu"
scenes = detector.predict(str(video_path), show_progressbar=True)

cap = cv2.VideoCapture(str(video_path))
for i, (start, end) in enumerate(scenes):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(f"scene_{i:04d}_start.png", frame)
cap.release()
```

---

## Configuration

### Device selection

- `SceneDetector()`:
  - uses **CUDA** if `torch.cuda.is_available()` else CPU
- `SceneDetector(torch.device("cpu"))`:
  - always uses **PyAV**
- `SceneDetector(torch.device("cuda"))`:
  - tries **NVDEC** first (if PyNvVideoCodec installed), falls back to **PyAV** on decode errors.

### Progress bars

- `show_progressbar=True` will enable `tqdm` for:
  - NVDEC window iteration (GPU backend)
  - frame extraction / window iteration (PyAV backend)

---

## Logging

The library uses the standard Python `logging` module. To see logs:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## License

This project inherits the MIT License from the original TransNetV2 repository. See `LICENSE` for details.
