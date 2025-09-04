# Whisper Speech Transcriber

[中文文档](README.md)

Offline transcription tool built with **faster-whisper (CTranslate2)** or **whisper.cpp (ggml)** plus **Tkinter**:
- Select audio/video files and convert them to **SRT** subtitles or **TXT** text with one click
- Automatically detects and prefers **GPU (CUDA)**; GUI provides **auto/cpu/gpu** options.
  If GPU initialization fails when forced, a warning is shown and execution stops; CPU mode is recommended in this case.
  In auto mode, the program falls back to CPU when GPU memory is insufficient and logs a warning.
- When using **CPU**, choose **int8/int16/int32** precision (default int8).
  For **GPU**, choose **float16/float32** (default float16); this option is disabled in auto mode.
  If precision is not specified, it is selected automatically and falls back when necessary.
  Incompatible precision settings will raise errors directly.
  Some CTranslate2 builds only support `int8`/`float32`; selecting unsupported precision (e.g. `int16`) will produce an error.
- CTranslate2 models have fixed precision at conversion time (e.g. directory names containing `int8`/`int16`).
  Inference must use the same or higher precision; for example, a `ct2int16` model cannot run with `int8` precision.
- Chinese is supported by default (`language="zh"`) with optional auto-detection.
- Optional **Beam Search** (`beam_width=10`, `n_best=5`) explores multiple hypotheses to improve accuracy.

<img width="1164" height="748" alt="image" src="https://github.com/user-attachments/assets/20215de2-fbf4-447b-b811-c6b01698d8a8" />

## Motivation

I initially relied on **WhisperDesktop** for transcription, yet it has not been updated for a long time and fails to work with the latest **Whisper v3** and **Whisper v3-turbo** models. To keep pace with new releases, I built this open-source assistant on top of **faster-whisper** and **whisper.cpp**.

## Quick Start

```bash
pip install -r requirements.txt
python whisper_assistant.py
```

The dependencies already include `pywhispercpp` to support ggml models. You can switch between **CTranslate2** and **ggml** backends in the GUI; when choosing ggml, specify the `.bin`/`.gguf` model file.

Checking **Beam Search** explores multiple hypotheses simultaneously to improve recognition accuracy.

> Windows users can place `ffmpeg.exe` in the program directory (alongside `whisper_assistant.py`) or ensure it is in the system PATH; Linux users should install `ffmpeg` via their package manager.

### GPU Dependencies (Optional)

The default `requirements.txt` only contains the minimal dependencies for CPU inference. To enable GPU (CUDA) acceleration, install:

```bash
pip install ctranslate2>=3.24.0
# Select the PyTorch build matching your CUDA version, e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> Native Windows 11 currently only provides the CPU version of `ctranslate2`. For GPU acceleration, consider running the above in [WSL2](https://learn.microsoft.com/windows/wsl/); if you only run on Windows, you may ignore GPU dependencies.

### Model Placement
- **CTranslate2**: Rename the converted model directory to `belle-whisper-large-v3-turbo-ct2i8f16` and place it under the `models/` subdirectory (loaded from there by default), or manually select the directory in the GUI.
- **ggml**: Download the `ggml`/`gguf` model file (e.g. `ggml-base.bin`) and select it directly in the GUI.

### CUDA (Optional)
- With an NVIDIA GPU and **CUDA 12.x + cuDNN 8**, the program automatically uses the GPU (`device=cuda, compute_type=float16`); otherwise, it falls back to CPU (`int8`).
- For the ggml backend, if `pywhispercpp` was installed with `GGML_CUDA=1`, it will also attempt to use the GPU automatically.

## Package into exe (Windows)
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole --name "WhisperTranscriber" whisper_assistant.py \
  --add-binary "ffmpeg.exe;."
```
Distribute the generated `dist/WhisperTranscriber.exe` to users (along with the model directory and `ffmpeg.exe`).

### Create Windows Installer (Optional)
1. Generate `dist/WhisperTranscriber.exe` with PyInstaller as above.
2. Install [Inno Setup](https://jrsoftware.org/).
3. Edit or reuse the `installer/WhisperTranscriber.iss` script in this repository, ensuring paths match `dist/`, `ffmpeg/`, and the model directory.
4. Run on the command line:
   ```powershell
   iscc installer\WhisperTranscriber.iss
   ```
   This creates a `WhisperTranscriberSetup.exe` installer under the `installer/` directory.
5. Distribute the installer together with models and other resources.

## Suggested Directory Layout
```
WhisperSpeechAssistant/
├─ whisper_assistant.py
├─ ffmpeg.exe
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ models/
   ├─ whisper-large-v3-turbo-ct2-i8f16/  # CTranslate2 model directory
   │   └─ ...
   └─ ggml-base.bin                      # example ggml model file
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

