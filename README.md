# Whisper 语音识别助手 · Whisper Speech Transcriber

基于 **faster-whisper (CTranslate2)** 或 **whisper.cpp (ggml)** + **Tkinter** 的离线转写工具：
- 选择音频/视频文件，一键转成 **SRT** 字幕或 **TXT** 文本
- 默认自动检测并优先使用 **GPU(CUDA)**，界面提供 **auto/cpu/gpu** 选项；
  若强制选择 GPU 但初始化失败会提示并停止，建议改用 **CPU**
- 选择 **CPU** 设备时，可在旁边选择 **int8/int16/int32** 精度（默认 int8）；
  选择 **GPU** 时，可选 **float16/float32**（默认 float16）；**auto** 模式下该选项不可用；
  未指定精度时，程序会自动选择并在必要时回退；若手动选择的精度与模型或设备不兼容，将直接报错并在日志中提示原因；
  某些 CTranslate2 发行版仅包含 `int8`/`float32` 支持，选择 `int16` 等未编译精度时会报错
- CTranslate2 模型在转换时已固定精度（如目录名含 `int8`/`int16` 等），
  推理时需使用相同或更高精度；例如 `ct2int16` 模型无法以 `int8` 推理；
  程序仅根据模型目录名推测其精度，`config.json` 中不包含量化信息
- 支持中文（默认 `language="zh"`），可切换自动检测
- 可选 **Beam Search**（默认 `beam_width=10`, `n_best=5`），通过同时探索多条假设路径提升识别正确率

## 项目背景 · Motivation

我最初使用 **WhisperDesktop** 作为转写工具，但该项目已长期停止维护，无法很好地兼容最新的 **Whisper v3** 和 **Whisper v3-turbo** 模型。为获得持续升级的体验，我基于 **faster-whisper** 与 **whisper.cpp** 重新实现了这个开源助手。

I initially relied on **WhisperDesktop** for transcription, yet it has not been updated for a long time and fails to work with the latest **Whisper v3** and **Whisper v3-turbo** models. To keep pace with new releases, I built this open-source assistant on top of **faster-whisper** and **whisper.cpp**.

## 快速开始

```bash
pip install -r requirements.txt
python whisper_assistant.py
```

依赖中已包含 `pywhispercpp` 以支持 ggml 模型。界面中可在 **CTranslate2** 与 **ggml** 两种后端之间切换，选择 ggml 时需要指定 `.bin`/`.gguf` 模型文件。

界面中勾选 **Beam Search** 可同时探索多条假设路径，从而提升识别准确率。

> Windows 用户可将 `ffmpeg.exe` 放在程序目录（与 `whisper_assistant.py` 同级）或确保其已在系统 PATH 中；Linux 用户通过系统包管理器安装 `ffmpeg`。

### GPU 依赖（可选）

默认 `requirements.txt` 仅包含 CPU 推理所需的最小依赖。
若希望启用 GPU (CUDA) 加速，请额外安装：

```bash
pip install ctranslate2>=3.24.0
# 根据你的 CUDA 版本选择对应的 PyTorch 发行版，例如 CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> Windows 11 原生环境目前仅提供 CPU 版本的 `ctranslate2`，如需 GPU 加速建议在 [WSL2](https://learn.microsoft.com/windows/wsl/) 中安装以上依赖；若仅在 Windows 上运行，可忽略 GPU 依赖。

### 模型放置
- **CTranslate2**：将转换好的模型目录命名为 `belle-whisper-large-v3-turbo-ct2i8f16` 并放在 `models/` 子目录（默认从该目录加载），或在界面中手动选择目录。
- **ggml**：下载 `ggml`/`gguf` 模型文件（例如 `ggml-base.bin`），在界面中直接选择该文件即可。

### CUDA（可选）
- 有 NVIDIA 显卡并安装 **CUDA 12.x + cuDNN 8** 时，程序会自动使用 GPU（`device=cuda, compute_type=float16`），否则回退到 CPU（`int8`）。
- ggml 后端若在安装 `pywhispercpp` 时启用了 `GGML_CUDA=1`，同样会自动尝试使用 GPU。

## 打包为 exe（Windows）
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole --name "WhisperTranscriber" whisper_assistant.py \
  --add-binary "ffmpeg.exe;."
```
将生成的 `dist/WhisperTranscriber.exe` 分发给用户即可（同时分发模型目录和 `ffmpeg.exe`）。

### 创建 Windows 安装包（可选）
1. 先按上文使用 PyInstaller 生成 `dist/WhisperTranscriber.exe`。
2. 安装 [Inno Setup](https://jrsoftware.org/)。
3. 编辑或直接使用仓库中的 `installer/WhisperTranscriber.iss` 脚本，确保路径与 `dist/`、`ffmpeg/` 和模型目录匹配。
4. 在命令行运行：
   ```powershell
   iscc installer\WhisperTranscriber.iss
   ```
   在 `installer/` 目录下会生成 `WhisperTranscriberSetup.exe` 安装包。
5. 将安装包与模型等资源一并分发给用户。

## 目录建议
```
WhisperSpeechAssistant/
├─ whisper_assistant.py
├─ ffmpeg.exe
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ models/
   ├─ belle-whisper-large-v3-turbo-ct2i8f16/  # CTranslate2 模型目录
   │   └─ ...
   └─ ggml-base.bin                           # ggml 模型文件示例
```

## 许可证
本项目使用 MIT 许可证，详见 [LICENSE](LICENSE)。
