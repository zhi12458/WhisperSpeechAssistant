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

> Windows 用户可将 `ffmpeg.exe` 放在程序同目录的 `ffmpeg/` 文件夹中；Linux 用户通过系统包管理器安装 `ffmpeg`。

### 模型放置
- 将你转换好的 CTranslate2 模型目录命名为：`belle-whisper-large-v3-turbo-ct2f16`，并放在与程序同级目录。  
- 或在界面中手动选择模型目录。

### CUDA（可选）
- 有 NVIDIA 显卡并安装 **CUDA 12.x + cuDNN 8** 时，程序会自动使用 GPU（`device=cuda, compute_type=float16`），否则回退到 CPU（`int8`）。

## 打包为 exe（Windows）
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole --name "WhisperTranscriber" whisper_assistant.py \
  --add-binary "ffmpeg/ffmpeg.exe;ffmpeg"
```
将生成的 `dist/WhisperTranscriber.exe` 分发给用户即可（同时分发模型目录和 `ffmpeg/ffmpeg.exe`）。

## 目录建议
```
WhisperSpeechAssistant/
├─ whisper_assistant.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ ffmpeg/
│  └─ ffmpeg.exe          # 可选
└─ belle-whisper-large-v3-turbo-ct2f16/  # 你的 CTranslate2 模型目录（不建议直接上传到 GitHub）
```

## 许可证
本项目使用 MIT 许可证，详见 [LICENSE](LICENSE)。
