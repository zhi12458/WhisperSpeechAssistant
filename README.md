# Whisper 语音识别助手 · Whisper Speech Transcriber

基于 **faster-whisper (CTranslate2)** 或 **whisper.cpp (ggml)** + **Tkinter** 的离线转写工具：
- 选择音频/视频文件，一键转成 **SRT** 字幕或 **TXT** 文本
- 默认自动检测并优先使用 **GPU(CUDA)**，界面提供 **auto/cpu/gpu** 选项；若强制选择 GPU 但初始化失败会提示并停止，建议改用 **CPU**
- 支持中文（默认 `language="zh"`），可切换自动检测

## 快速开始

```bash
pip install -r requirements.txt
python whisper_assistant.py
```

依赖中已包含 `pywhispercpp` 以支持 ggml 模型。界面中可在 **CTranslate2** 与 **ggml** 两种后端之间切换，选择 ggml 时需要指定 `.bin`/`.gguf` 模型文件。

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
 - **CTranslate2**：将转换好的模型目录命名为 `belle-whisper-large-v3-turbo-ct2-int16` 并放在 `models/` 子目录（默认从该目录加载），或在界面中手动选择目录。
- **ggml**：下载 `ggml`/`gguf` 模型文件（例如 `ggml-base.bin`），在界面中直接选择该文件即可。

### CUDA（可选）
 - 有 NVIDIA 显卡并安装 **CUDA 12.x + cuDNN 8** 时，程序会自动使用 GPU（`device=cuda, compute_type=float16`），否则回退到 CPU（`int16`）。
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
    ├─ belle-whisper-large-v3-turbo-ct2-int16/  # CTranslate2 模型目录
   │   └─ ...
   └─ ggml-base.bin                           # ggml 模型文件示例
```

## 许可证
本项目使用 MIT 许可证，详见 [LICENSE](LICENSE)。
