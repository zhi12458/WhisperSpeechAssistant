# whisper_assistant.py
# -*- coding: utf-8 -*-
import os
import sys
import datetime
import traceback
import threading
import queue
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json

# pip install faster-whisper
from faster_whisper import WhisperModel

APP_TITLE = "Whisper 语音识别助手 (Whisper Speech Transcriber)"

def ensure_ffmpeg_on_path():
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    if shutil.which("ffmpeg"):
        return
    local_ffmpeg = os.path.join(exe_dir, "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")
    if os.path.isfile(local_ffmpeg):
        os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ffmpeg"):
        return
    msg = "未检测到 ffmpeg 可执行文件，请安装或将其放在程序目录。"
    try:
        messagebox.showwarning("FFmpeg 未找到", msg)
    except Exception:
        pass
    raise FileNotFoundError(msg)

ensure_ffmpeg_on_path()

def format_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    td = datetime.timedelta(seconds=float(seconds))
    total_ms = int(td.total_seconds() * 1000)
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60_000) % 60
    h = total_ms // 3_600_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(segments, outfile: str):
    with open(outfile, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            text = (seg.text or "").strip()
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

def write_txt(segments, outfile: str):
    with open(outfile, "w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                f.write(text + "\\n")

def open_in_explorer(path: str):
    try:
        folder = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else os.path.abspath(path)
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore
        elif sys.platform == "darwin":
            subprocess.run(["open", folder], check=False)
        else:
            subprocess.run(["xdg-open", folder], check=False)
    except Exception:
        pass

def is_ggml_model(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith((".bin", ".ggml", ".gguf"))

def get_media_duration(media_path: str) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                media_path,
            ],
            stderr=subprocess.STDOUT,
        )
        return float(out.strip())
    except Exception:
        return None

def pick_device_and_compute_type():
    if os.environ.get("WHISPER_FORCE_CPU") == "1":
        return "cpu", "int8"
    try:
        import ctranslate2 as c2  # type: ignore
        get_cnt = getattr(c2, "get_cuda_device_count", None)
        if callable(get_cnt) and get_cnt() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"

_model_cache = {}


def detect_model_device_compute(model_path: str):
    """Try to infer preferred (device, compute_type) from model quantization."""
    quant = None
    config_path = os.path.join(model_path, "config.json")
    _QUANT_SYNONYMS = {
        "int8_float16": ["int8_float16", "i8f16"],
        "float16": ["float16", "f16"],
        "int16": ["int16", "i16"],
        "int8": ["int8", "i8"],
    }
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                q = data.get("quantization")
                if isinstance(q, str):
                    q = q.lower()
                    for k, syns in _QUANT_SYNONYMS.items():
                        if q in syns:
                            quant = k
                            break
        except Exception:
            pass
    if not quant:
        name = os.path.basename(os.path.abspath(model_path)).lower()
        for k, syns in _QUANT_SYNONYMS.items():
            for s in sorted(syns, key=len, reverse=True):
                if s in name:
                    quant = k
                    break
            if quant:
                break
    if quant:
        cuda_available = False
        try:
            import ctranslate2 as c2  # type: ignore
            get_cnt = getattr(c2, "get_cuda_device_count", None)
            if callable(get_cnt) and get_cnt() > 0:
                cuda_available = True
        except Exception:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    cuda_available = True
            except Exception:
                pass
        if quant == "int8":
            return "cpu", "int8"
        if quant == "int16":
            return "cpu", "int16"
        if quant == "float16" and cuda_available:
            return "cuda", "float16"
        if quant == "int8_float16":
            if cuda_available:
                return "cuda", "float16"
            return "cpu", "int8"
    return None

def load_model(model_path: str, backend: str):
    if backend == "ggml":
        key = ("ggml", model_path)
        if key in _model_cache:
            return _model_cache[key], "cpu", "ggml"
        from pywhispercpp.model import Model  # type: ignore

        n_threads = os.cpu_count() or 4
        model = Model(model_path, n_threads=n_threads)
        _model_cache[key] = model
        return model, "cpu", "ggml"
    preferred = detect_model_device_compute(model_path)
    if preferred:
        device, first_ct = preferred
    else:
        device, first_ct = pick_device_and_compute_type()
    if device == "cuda":
        fallbacks = [first_ct, "float32", "int8"]
    else:
        fallbacks = [first_ct, "int16", "float32", "float16"]
    last_err = None
    for ct in fallbacks:
        key = (model_path, device, ct)
        if key in _model_cache:
            return _model_cache[key], device, ct
        try:
            model = WhisperModel(model_path, device=device, compute_type=ct)
            _model_cache[key] = model
            return model, device, ct
        except Exception as e:
            last_err = e
    try:
        model = WhisperModel(model_path, device="cpu", compute_type="float32")
        return model, "cpu", "float32"
    except Exception:
        raise last_err if last_err else RuntimeError("模型加载失败")

def transcribe_with_progress(model_path: str, media_path: str, fmt: str, language: str, backend: str, logger, progress_cb):
    if not os.path.isfile(media_path):
        raise FileNotFoundError(f"未找到文件：{media_path}")
    if is_ggml_model(model_path):
        backend = "ggml"
    model, device, compute_type = load_model(model_path, backend)
    if backend == "ggml":
        logger(f"[INFO] Using ggml backend")
        duration = get_media_duration(media_path)
        if not duration or duration <= 0:
            progress_cb(("mode", "indeterminate"))
        else:
            progress_cb(("mode", "determinate"))
            progress_cb(0)
        seg_list = []
        last_p = 0

        def cb(seg):
            nonlocal last_p
            seg.start = getattr(seg, "t0", 0) / 100.0
            seg.end = getattr(seg, "t1", 0) / 100.0
            seg_list.append(seg)
            end_t = seg.end
            if duration and duration > 0:
                p = int(min(100, max(0, (end_t / duration) * 100)))
                if p > last_p:
                    last_p = p
                    progress_cb(p)
            logger(
                f"[SEG {len(seg_list)}] {format_timestamp(seg.start)} --> {format_timestamp(seg.end)} {seg.text.strip()}"
            )

        model.transcribe(media_path, language=(language or ""), new_segment_callback=cb, print_progress=False)
        progress_cb(100)
        segments = seg_list
    else:
        logger(f"[INFO] Using device={device}, compute_type={compute_type}")
        segments, info = model.transcribe(
            media_path,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5,
            word_timestamps=False,
        )
        duration = getattr(info, "duration", None)
        if not duration or duration <= 0:
            progress_cb(("mode", "indeterminate"))
        else:
            progress_cb(("mode", "determinate"))
            progress_cb(0)
        seg_list = []
        last_p = 0
        last_end = 0.0
        for i, seg in enumerate(segments, start=1):
            seg_list.append(seg)
            end_t = seg.end if seg.end is not None else last_end
            last_end = end_t
            if duration and duration > 0:
                p = int(min(100, max(0, (end_t / duration) * 100)))
                if p > last_p:
                    last_p = p
                    progress_cb(p)
            logger(
                f"[SEG {i}] {format_timestamp(seg.start)} --> {format_timestamp(seg.end)} {seg.text.strip()}"
            )
        progress_cb(100)
        segments = seg_list
    base, _ = os.path.splitext(media_path)
    outfile = base + (".srt" if fmt == "srt" else ".txt")
    if fmt == "srt":
        write_srt(segments, outfile)
    else:
        write_txt(segments, outfile)
    return outfile

class WhisperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("760x420")
        self.resizable(False, False)
        self.msg_queue = queue.Queue()
        self.worker_thread = None
        self.last_output = None
        exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
        default_model_dir = os.path.join(exe_dir, "models", "belle-whisper-large-v3-turbo-ct2i8f16")
        tk.Label(self, text="后端:").grid(row=0, column=0, sticky="w", padx=12, pady=8)
        self.backend_var = tk.StringVar(value="ct2")
        tk.Radiobutton(self, text="CTranslate2", variable=self.backend_var, value="ct2").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(self, text="ggml", variable=self.backend_var, value="ggml").grid(row=0, column=2, sticky="w")
        tk.Label(self, text="模型路径:").grid(row=1, column=0, sticky="w", padx=12, pady=8)
        self.model_entry = tk.Entry(self, width=62)
        self.model_entry.insert(0, default_model_dir)
        self.model_entry.grid(row=1, column=1, padx=6)
        tk.Button(self, text="选择", width=8, command=self.choose_model_path).grid(row=1, column=2, padx=6)
        tk.Label(self, text="音/视频文件:").grid(row=2, column=0, sticky="w", padx=12, pady=8)
        self.media_entry = tk.Entry(self, width=62)
        self.media_entry.grid(row=2, column=1, padx=6)
        tk.Button(self, text="选择", width=8, command=self.choose_media_file).grid(row=2, column=2, padx=6)
        tk.Label(self, text="输出格式:").grid(row=3, column=0, sticky="w", padx=12, pady=8)
        self.output_fmt = tk.StringVar(value="srt")
        tk.Radiobutton(self, text="SRT", variable=self.output_fmt, value="srt").grid(row=3, column=1, sticky="w")
        tk.Radiobutton(self, text="TXT", variable=self.output_fmt, value="txt").grid(row=3, column=1)
        tk.Label(self, text="识别语言:").grid(row=3, column=2, sticky="w")
        self.lang_combo = ttk.Combobox(self, values=["zh", "auto"], width=8, state="readonly")
        self.lang_combo.set("zh")
        self.lang_combo.grid(row=3, column=2, padx=70, sticky="e")
        tk.Label(self, text="进度:").grid(row=4, column=0, sticky="w", padx=12, pady=8)
        self.progress = ttk.Progressbar(self, orient="horizontal", length=560, mode="determinate", maximum=100)
        self.progress.grid(row=4, column=1, columnspan=2, padx=6, sticky="w")
        self.progress_pct = tk.Label(self, text="0%")
        self.progress_pct.grid(row=4, column=2, sticky="e", padx=10)
        self.start_button = tk.Button(self, text="开始转写", width=12, command=self.on_run)
        self.start_button.grid(row=5, column=1, pady=6, sticky="e")
        tk.Button(self, text="打开输出所在文件夹", width=18, command=self.on_open_folder).grid(row=5, column=2, pady=6, sticky="w")
        tk.Label(self, text="日志:").grid(row=6, column=0, sticky="nw", padx=12, pady=8)
        self.log_text = tk.Text(self, height=10, width=92, state="disabled")
        self.log_text.grid(row=6, column=1, columnspan=2, padx=6, pady=6)
        self.after(100, self.poll_queue)

    def choose_model_path(self):
        if self.backend_var.get() == "ggml":
            path = filedialog.askopenfilename(filetypes=[("ggml 模型", "*.bin *.ggml *.gguf"), ("所有文件", "*.*")])
        else:
            path = filedialog.askdirectory()
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)

    def choose_media_file(self):
        path = filedialog.askopenfilename(filetypes=[("媒体文件", "*.wav *.mp3 *.m4a *.flac *.mp4 *.mkv *.mov *.avi"), ("所有文件", "*.*")])
        if path:
            self.media_entry.delete(0, tk.END)
            self.media_entry.insert(0, path)

    def log(self, text: str):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def enqueue(self, msg):
        self.msg_queue.put(msg)

    def poll_queue(self):
        while True:
            try:
                msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(msg, tuple) and msg and msg[0] == "PROG":
                payload = msg[1]
                if isinstance(payload, tuple) and payload[0] == "mode":
                    mode = payload[1]
                    if mode == "indeterminate":
                        self.progress.configure(mode="indeterminate")
                        self.progress.start(80)
                        self.progress_pct.config(text="计算中...")
                    else:
                        self.progress.stop()
                        self.progress.configure(mode="determinate")
                        self.progress["value"] = 0
                        self.progress_pct.config(text="0%")
                else:
                    p = int(payload)
                    self.progress.configure(mode="determinate")
                    self.progress["value"] = p
                    self.progress_pct.config(text=f"{p}%")
            elif isinstance(msg, tuple) and msg and msg[0] == "DONE":
                ok, payload = msg[1], msg[2]
                self.progress.stop()
                if ok:
                    self.last_output = payload
                    self.log(f"✅ 完成：{payload}")
                    self.progress["value"] = 100
                    self.progress_pct.config(text="100%")
                    messagebox.showinfo("完成", f"输出文件：{payload}")
                else:
                    self.log("❌ 出错：")
                    self.log(payload)
                    messagebox.showerror("出错", payload)
                self.set_running(False)
            else:
                self.log(str(msg))
        self.after(120, self.poll_queue)

    def set_running(self, running: bool):
        self.start_button.config(state="disabled" if running else "normal")

    def on_open_folder(self):
        if self.last_output and os.path.exists(self.last_output):
            open_in_explorer(self.last_output)
        else:
            messagebox.showinfo("提示", "还没有生成任何文件。")

    def on_run(self):
        model_path = self.model_entry.get().strip()
        media_path = self.media_entry.get().strip()
        fmt = self.output_fmt.get()
        lang = self.lang_combo.get()
        backend = self.backend_var.get()
        if is_ggml_model(model_path):
            backend = "ggml"
        if backend == "ggml":
            if not os.path.isfile(model_path):
                messagebox.showerror("错误", "请选择 ggml 模型文件")
                return
        else:
            if not os.path.isdir(model_path):
                messagebox.showerror("错误", "模型目录不存在")
                return
        if not os.path.isfile(media_path):
            messagebox.showerror("错误", "请选择需要识别的音/视频文件")
            return
        self.log(f"开始转写：{media_path}")
        self.log(f"使用模型：{model_path} (后端：{backend})")
        self.log(f"输出格式：{fmt.upper()}，语言：{lang}")
        self.set_running(True)
        self.progress.configure(mode="determinate")
        self.progress["value"] = 0
        self.progress_pct.config(text="0%")

        def progress_cb(v):
            self.enqueue(("PROG", v))
        def worker():
            try:
                outfile = transcribe_with_progress(
                    model_path=model_path,
                    media_path=media_path,
                    fmt=fmt,
                    language=(None if lang == "auto" else lang),
                    backend=backend,
                    logger=self.enqueue,
                    progress_cb=progress_cb,
                )
                self.enqueue(("DONE", True, outfile))
            except Exception as e:
                err = f"{e}\n{traceback.format_exc()}"
                self.enqueue(("DONE", False, err))
        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    WhisperApp().mainloop()
