# Whisper 语音识别助手 · Whisper Speech Transcriber

基于 **faster-whisper (CTranslate2)** + **Tkinter** 的离线转写工具：
- 选择音频/视频文件，一键转成 **SRT** 字幕或 **TXT** 文本
- 自动检测 **GPU(CUDA)** 或 **CPU**，支持进度条
- 支持中文（默认 `language="zh"`），可切换自动检测

## 快速开始

```bash
pip install -r requirements.txt
python whisper_assistant.py
```

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
- 将你转换好的 CTranslate2 模型目录命名为：`belle-whisper-large-v3-turbo-ct2i8f16`，并放在 `models/` 子目录（默认会从 `models/belle-whisper-large-v3-turbo-ct2i8f16` 加载）。
- 或在界面中手动选择模型目录。

### CUDA（可选）
- 有 NVIDIA 显卡并安装 **CUDA 12.x + cuDNN 8** 时，程序会自动使用 GPU（`device=cuda, compute_type=float16`），否则回退到 CPU（`int8`）。

## 打包为 exe（Windows）
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole --name "WhisperTranscriber" whisper_assistant.py \
  --add-binary "ffmpeg.exe;."
```
将生成的 `dist/WhisperTranscriber.exe` 分发给用户即可（同时分发模型目录和 `ffmpeg.exe`）。

## 目录建议
```
WhisperSpeechAssistant/
├─ whisper_assistant.py
├─ ffmpeg.exe
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ models/
   └─ belle-whisper-large-v3-turbo-ct2i8f16/
      └─ ...  # 模型文件
```

## 许可证
本项目使用 MIT 许可证，详见 [LICENSE](LICENSE)。
