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

# pip install faster-whisper
from faster_whisper import WhisperModel
# ``load_audio`` was removed in recent versions of ``faster-whisper``.
# ``decode_audio`` is a drop-in replacement that returns the raw audio
# samples resampled to the requested rate (16 kHz by default).
# Import it here and alias it back to ``load_audio`` so the rest of the
# codebase can continue to call ``load_audio`` transparently.
from faster_whisper.audio import decode_audio as load_audio


class TranscriptionStopped(Exception):
    """Raised when the transcription is interrupted by the user."""
    pass

APP_TITLE = "Whisper 语音识别助手 (Whisper Speech Transcriber)"

# Default parameters for optional beam search decoding
DEFAULT_BEAM_WIDTH = 10
DEFAULT_N_BEST = 5
DEFAULT_TOP_K = 4
TS_PROB_THRESHOLD = 0.01
TS_PROB_SUM_THRESHOLD = 0.01


def sample_best(
    logits,
    timestamp_mask,
    top_k: int = DEFAULT_TOP_K,
    thold_pt: float = TS_PROB_THRESHOLD,
    thold_ptsum: float = TS_PROB_SUM_THRESHOLD,
):
    """Select a token using timestamp filtering and Top-k sampling.

    This helper performs a timestamp probability check on ``logits`` before
    choosing the final token from the ``top_k`` highest probability
    candidates.  The approach mirrors the ``sampleBest`` strategy in
    whisper.cpp and helps avoid returning extremely low-probability words.

    Args:
        logits: Sequence of raw logits for all tokens.
        timestamp_mask: Boolean mask indicating timestamp token positions.
        top_k: Number of candidate tokens to consider (default 4).
        thold_pt: Minimum individual timestamp token probability.
        thold_ptsum: Minimum combined timestamp token probability.

    Returns:
        The index of the selected token.
    """

    import numpy as np

    probs = np.exp(logits - np.max(logits))
    ts_probs = probs[timestamp_mask]
    ts_max = ts_probs.max() if ts_probs.size > 0 else 0.0
    if ts_max < thold_pt or ts_probs.sum() < thold_ptsum:
        probs[timestamp_mask] = 0.0
    top_indices = np.argsort(probs)[-top_k:]
    best_idx = top_indices[np.argmax(probs[top_indices])]
    return int(best_idx)

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
                f.write(text + "\n")

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

def pick_device_and_compute_type(mode: str = "auto"):
    warn = None
    if mode == "cpu" or os.environ.get("WHISPER_FORCE_CPU") == "1":
        return "cpu", "int8", warn
    if mode in ("gpu", "auto"):
        has_cuda = False
        try:
            import ctranslate2 as c2  # type: ignore
            get_cnt = getattr(c2, "get_cuda_device_count", None)
            if callable(get_cnt) and get_cnt() > 0:
                has_cuda = True
                get_supported = getattr(c2, "get_supported_compute_types", None)
                if callable(get_supported):
                    try:
                        supported = get_supported("cuda")
                    except Exception:
                        supported = []
                    if "float16" in supported:
                        return "cuda", "float16", warn
                    if "float32" in supported:
                        return "cuda", "float32", warn
                    if supported:
                        return "cuda", supported[0], warn
                return "cuda", "float16", warn
        except Exception:
            pass
        if not has_cuda:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    return "cuda", "float16", warn
            except Exception:
                pass
            warn = "未检测到可用 GPU"
    return "cpu", "int8", warn

_model_cache = {}


def cleanup_model(model, logger=lambda msg: None):
    """Remove model from cache and release CUDA memory."""
    # Drop any cached references to the model
    for k, v in list(_model_cache.items()):
        if v is model:
            _model_cache.pop(k, None)
    try:
        logger("[DEBUG] releasing model")
        del model
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.cuda.empty_cache()
        logger("[DEBUG] CUDA cache emptied")
    except Exception:
        logger("[DEBUG] torch.cuda.empty_cache() unavailable")


def guess_model_precision(model_path: str) -> str | None:
    """Best-effort detection of a CTranslate2 model's quantization.

    ``config.json`` does not record the quantization of converted models, so the
    detection relies solely on the directory name (e.g. ``ct2int8``/``ct2int16``).
    """
    # ``os.path.basename`` returns an empty string when ``model_path`` ends with
    # a path separator.  Normalize first so trailing ``/`` or ``\\`` don't hide
    # the directory name (which often embeds the precision, e.g. ``ct2int8``).
    lower = os.path.basename(os.path.normpath(model_path)).lower()
    for ct in ("int8", "int16", "int32", "float16", "float32"):
        if ct in lower:
            return ct
    # handle names like "i8f16" where the activation precision follows an "f"
    import re
    m = re.search(r"i\d+f(16|32)", lower)
    if m:
        return f"float{m.group(1)}"
    return None

def load_model(
    model_path: str,
    backend: str,
    device_mode: str = "auto",
    compute_type: str | None = None,
):
    warn_msg = None
    ct_err = None
    if backend == "ggml":
        from pywhispercpp.model import Model  # type: ignore

        n_threads = os.cpu_count() or 4
        params = {}
        device = "cpu"
        if device_mode == "gpu":
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    params["n_gpu_layers"] = 99
                    device = "cuda"
                else:
                    raise RuntimeError
            except Exception:
                raise RuntimeError("GPU 初始化失败，请切换到 CPU 模式")
        elif device_mode == "auto":
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    params["n_gpu_layers"] = 99
                    device = "cuda"
            except Exception:
                pass
        key = ("ggml", model_path, device)
        if key in _model_cache:
            return _model_cache[key], device, "ggml", warn_msg, None
        try:
            model = Model(model_path, n_threads=n_threads, **params)
        except Exception:
            if device == "cuda":
                if device_mode == "gpu":
                    raise RuntimeError("GPU 初始化失败，请切换到 CPU 模式")
                warn_msg = "GPU 初始化失败，已切回 CPU 模式"
                params = {}
                device = "cpu"
                key = ("ggml", model_path, device)
                model = Model(model_path, n_threads=n_threads)
            else:
                raise
        _model_cache[key] = model
        return model, device, "ggml", warn_msg, None
    device, default_ct, device_warn = pick_device_and_compute_type(device_mode)
    if device_warn:
        warn_msg = device_warn

    # If the user explicitly chose both device and precision, try to honor it
    # but fall back to CPU+int8 if the GPU float16 request fails.
    if device_mode in ("cpu", "gpu") and compute_type is not None:
        if device_mode == "gpu" and device != "cuda":
            raise RuntimeError(device_warn or "GPU 初始化失败，请切换到 CPU 模式")
        key = (model_path, device, compute_type)
        if key in _model_cache:
            return _model_cache[key], device, compute_type, warn_msg, None
        try:
            model = WhisperModel(model_path, device=device, compute_type=compute_type)
            _model_cache[key] = model
            return model, device, compute_type, warn_msg, None
        except Exception as e:
            if device_mode == "gpu" and compute_type == "float16":
                err = str(e)
                lower = err.lower()
                if (
                    "out of memory" in lower
                    or "insufficient memory" in lower
                    or "failed to allocate" in lower
                    or "do not support efficient float16" in lower
                ):
                    warn_msg = "显存不足，已切回 CPU + int8 模式"
                    ct_err = (
                        "CUDA 显存不足"
                        if "do not support efficient float16" in lower
                        else err
                    )
                else:
                    warn_msg = "GPU 初始化失败，已切回 CPU + int8 模式"
                    ct_err = err
                device = "cpu"
                compute_type = "int8"
                key = (model_path, device, compute_type)
                if key in _model_cache:
                    return _model_cache[key], device, compute_type, warn_msg, ct_err
                model = WhisperModel(model_path, device=device, compute_type=compute_type)
                _model_cache[key] = model
                return model, device, compute_type, warn_msg, ct_err
            else:
                raise

    model_ct = None if backend == "ggml" else guess_model_precision(model_path)
    first_ct = compute_type or model_ct or default_ct

    if device_mode == "gpu" and device != "cuda":
        raise RuntimeError(device_warn or "GPU 初始化失败，请切换到 CPU 模式")

    fallbacks = [first_ct]
    if device == "cuda":
        if first_ct != "float16":
            for ct in ["float16", "float32", "int8"]:
                if ct not in fallbacks:
                    fallbacks.append(ct)
    else:
        for ct in ["int8", "int16", "int32", "float32", "float16"]:
            if ct not in fallbacks:
                fallbacks.append(ct)
    last_err = None
    for ct in fallbacks:
        key = (model_path, device, ct)
        if key in _model_cache:
            return _model_cache[key], device, ct, warn_msg, (None if ct == first_ct else ct_err)
        try:
            model = WhisperModel(model_path, device=device, compute_type=ct)
            _model_cache[key] = model
            return model, device, ct, warn_msg, (None if ct == first_ct else ct_err)
        except Exception as e:
            last_err = e
            if ct == first_ct:
                ct_err = str(e)
    if device == "cuda":
        if device_mode == "gpu":
            raise RuntimeError("GPU 初始化失败，请切换到 CPU 模式")
        if ct_err:
            lower = ct_err.lower()
            if (
                "out of memory" in lower
                or "insufficient memory" in lower
                or "failed to allocate" in lower
                or "do not support efficient float16" in lower
            ):
                warn_msg = "显存不足，已切回 CPU + int8 模式"
                ct_err = (
                    "CUDA 显存不足"
                    if "do not support efficient float16" in lower
                    else ct_err
                )
            else:
                warn_msg = "GPU 初始化失败，已切回 CPU + int8 模式"
        else:
            warn_msg = "GPU 初始化失败，已切回 CPU + int8 模式"
        device = "cpu"
        last_err = None
        for ct in ["int8", "int16", "int32", "float32", "float16"]:
            key = (model_path, device, ct)
            if key in _model_cache:
                return _model_cache[key], device, ct, warn_msg, ct_err
            try:
                model = WhisperModel(model_path, device=device, compute_type=ct)
                _model_cache[key] = model
                return model, device, ct, warn_msg, ct_err
            except Exception as e:
                last_err = e
                if ct == first_ct:
                    ct_err = str(e)
    if device_mode == "auto":
        try:
            model = WhisperModel(model_path, device="cpu", compute_type="float32")
            return model, "cpu", "float32", warn_msg, (None if "float32" == first_ct else ct_err)
        except Exception:
            raise last_err if last_err else RuntimeError("模型加载失败")
    raise last_err if last_err else RuntimeError("模型加载失败")


def run_full_transcribe(
    model,
    media_path,
    language,
    logger,
    progress_cb,
    stop_event,
    word_timestamps: bool = False,
    max_len: int | None = None,
    max_tokens: int | None = None,
    use_context: bool = False,
    beam_search: bool = False,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    n_best: int = DEFAULT_N_BEST,
    overlap: float = 1.0,
):
    """
    Process the entire audio in ~30s windows, accumulating segments.
    This mimics the "runFull" strategy where the whole file is read
    once and recognition happens on internal chunks that are later
    concatenated.
    """
    audio = load_audio(media_path)
    sample_rate = 16000
    total = len(audio)
    if total <= 0:
        return []
    progress_cb(("mode", "determinate"))
    progress_cb(0)
    chunk_samples = sample_rate * 30
    overlap_samples = int(overlap * sample_rate)
    stride = chunk_samples - overlap_samples if chunk_samples > overlap_samples else chunk_samples
    segments = []
    last_end = 0.0
    last_p = 0
    token_history = []
    prev_tokens: list[int] = []
    n_max_text_ctx = getattr(model, "max_length", 0)
    max_prompt_tokens = n_max_text_ctx // 2 if n_max_text_ctx else 0
    for start in range(0, total, stride):
        end = min(total, start + chunk_samples)
        if stop_event and stop_event.is_set():
            raise TranscriptionStopped()
        chunk = audio[start:end]
        kwargs = {"language": language, "word_timestamps": word_timestamps, "vad_filter": True}
        if beam_search:
            kwargs["beam_size"] = beam_width
            kwargs["best_of"] = n_best
        else:
            kwargs["beam_size"] = 1
            kwargs["best_of"] = DEFAULT_TOP_K
        if use_context and prev_tokens:
            if max_prompt_tokens:
                kwargs["initial_prompt"] = prev_tokens[-max_prompt_tokens:]
            else:
                kwargs["initial_prompt"] = prev_tokens
        if max_len is not None:
            kwargs["max_len"] = max_len
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        sub_segments, _ = model.transcribe(audio=chunk, **kwargs)
        current_tokens = []
        offset = start / sample_rate
        for seg in sub_segments:
            seg.start = (seg.start or 0.0) + offset
            seg.end = (seg.end or 0.0) + offset
            if seg.end > last_end - 0.2:
                if seg.start < last_end:
                    seg.start = last_end
                segments.append(seg)
                logger(
                    f"[SEG {len(segments)}] {format_timestamp(seg.start)} --> {format_timestamp(seg.end)} {seg.text.strip()}"
                )
                if use_context:
                    current_tokens.extend(seg.tokens)
                last_end = seg.end
        if use_context:
            prev_tokens.extend(current_tokens)
            if max_prompt_tokens and len(prev_tokens) > max_prompt_tokens:
                prev_tokens = prev_tokens[-max_prompt_tokens:]
            token_history.append(current_tokens)
        p = int(min(100, (end / total) * 100))
        if p > last_p:
            last_p = p
            progress_cb(p)
    progress_cb(100)
    return segments, token_history

def transcribe_with_progress(
    model_path: str,
    media_path: str,
    fmt: str,
    language: str,
    backend: str,
    logger,
    progress_cb,
    stop_event=None,
    device_mode: str = "auto",
    compute_type: str | None = None,
    word_timestamps: bool = False,
    max_len: int | None = None,
    max_tokens: int | None = None,
    use_context: bool = False,
    beam_search: bool = False,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    n_best: int = DEFAULT_N_BEST,
):
    if not os.path.isfile(media_path):
        raise FileNotFoundError(f"未找到文件：{media_path}")
    if is_ggml_model(model_path):
        backend = "ggml"
    model_prec = None if backend == "ggml" else guess_model_precision(model_path)
    requested_ct = compute_type

    def load_and_log(dev_mode: str):
        nonlocal model_prec, requested_ct
        default_ct = pick_device_and_compute_type(dev_mode)[1]
        first_ct = requested_ct or model_prec or default_ct
        mdl, dev, ct, warn, ct_err = load_model(
            model_path, backend, device_mode=dev_mode, compute_type=requested_ct
        )
        if first_ct != ct:
            if requested_ct:
                msg = f"Requested compute_type={requested_ct} but using {ct}"
            elif model_prec:
                msg = f"Model quantized to {model_prec} but using {ct}"
            else:
                msg = f"Auto-selected compute_type={first_ct} but using {ct}"
            if ct_err is not None:
                if str(ct_err).strip():
                    msg += f": {ct_err}"
                else:
                    msg += ": requested precision not supported by backend"
            logger(f"[WARN] {msg}")
        elif ct_err is not None:
            logger(f"[WARN] {ct_err}")
        if warn:
            logger(f"[WARN] {warn}")
            if "未检测到可用 GPU" not in warn:
                try:
                    messagebox.showwarning("GPU 初始化失败", warn)
                except Exception:
                    pass
        return mdl, dev, ct

    model = None
    try:
        model, device, compute_type = load_and_log(device_mode)
        logger(f"[DEBUG] model loaded on {device} with compute_type={compute_type}")
        if word_timestamps and max_len is None:
            max_len = 40
            logger(f"[INFO] word_timestamps enabled, set max_len={max_len}")
        if backend == "ggml":
            logger(f"[INFO] Using ggml backend (device={device})")
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
                if stop_event and stop_event.is_set():
                    raise TranscriptionStopped()

            kwargs = {
                "language": (language or ""),
                "new_segment_callback": cb,
                "print_progress": False,
                "greedy": {"best_of": DEFAULT_TOP_K},
                "thold_pt": TS_PROB_THRESHOLD,
                "thold_ptsum": TS_PROB_SUM_THRESHOLD,
            }
            if beam_search:
                kwargs["beam_search"] = {"beam_size": beam_width, "patience": -1.0}
            if word_timestamps:
                kwargs["word_timestamps"] = True
            if max_len is not None:
                kwargs["max_len"] = max_len
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            model.transcribe(media_path, **kwargs)
            progress_cb(100)
            segments = seg_list
        else:
            logger(f"[INFO] Using device={device}, compute_type={compute_type}")
            try:
                segments, _ = run_full_transcribe(
                    model,
                    media_path,
                    language,
                    logger,
                    progress_cb,
                    stop_event,
                    word_timestamps=word_timestamps,
                    max_len=max_len,
                    max_tokens=max_tokens,
                    use_context=use_context,
                    beam_search=beam_search,
                    beam_width=beam_width,
                    n_best=n_best,
                )
            except RuntimeError as e:
                if (
                    device_mode == "auto"
                    and device == "cuda"
                    and "out of memory" in str(e).lower()
                ):
                    logger("[WARN] CUDA out of memory, retrying on CPU")
                    progress_cb(0)
                    model, device, compute_type = load_and_log("cpu")
                    logger(
                        f"[INFO] Using device={device}, compute_type={compute_type}"
                    )
                    segments, _ = run_full_transcribe(
                        model,
                        media_path,
                        language,
                        logger,
                        progress_cb,
                        stop_event,
                        word_timestamps=word_timestamps,
                        max_len=max_len,
                        max_tokens=max_tokens,
                        use_context=use_context,
                        beam_search=beam_search,
                        beam_width=beam_width,
                        n_best=n_best,
                    )
                else:
                    raise
        base, _ = os.path.splitext(media_path)
        outfile = base + (".srt" if fmt == "srt" else ".txt")
        if fmt == "srt":
            write_srt(segments, outfile)
        else:
            write_txt(segments, outfile)
        logger("[DEBUG] transcription finished")
        return outfile
    finally:
        if model is not None:
            cleanup_model(model, logger)

class WhisperApp(tk.Tk):
    def __init__(self, use_context: bool = False):
        super().__init__()
        self.title(APP_TITLE)
        # slightly taller default window and allow resizing so controls stay visible
        self.geometry("760x460")
        self.resizable(True, True)
        # allow widgets to grow/shrink with window resizing
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(8, weight=1)
        self.msg_queue = queue.Queue()
        self.worker_thread = None
        self.last_output = None
        exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
        default_model_dir = os.path.join(exe_dir, "models", "belle-whisper-large-v3-turbo-ct2-i8f16")
        tk.Label(self, text="后端:").grid(row=0, column=0, sticky="w", padx=12, pady=8)
        self.backend_var = tk.StringVar(value="ct2")
        tk.Radiobutton(self, text="CTranslate2", variable=self.backend_var, value="ct2").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(self, text="ggml", variable=self.backend_var, value="ggml").grid(row=0, column=2, sticky="w")

        tk.Label(self, text="设备:").grid(row=1, column=0, sticky="w", padx=12)
        self.device_var = tk.StringVar(value="auto")
        self.device_combo = ttk.Combobox(
            self,
            textvariable=self.device_var,
            values=["auto", "cpu", "gpu"],
            width=8,
            state="readonly",
        )
        self.device_combo.current(0)
        self.device_combo.grid(row=1, column=1, sticky="w")
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        tk.Label(self, text="精度:").grid(row=1, column=2, sticky="w", padx=12)
        self.compute_type_var = tk.StringVar()
        self.compute_type_combo = ttk.Combobox(
            self,
            textvariable=self.compute_type_var,
            width=8,
            state="disabled",
        )
        self.compute_type_combo.grid(row=1, column=3, sticky="w")
        self.on_device_change()

        tk.Label(self, text="模型路径:").grid(row=2, column=0, sticky="w", padx=12, pady=8)
        self.model_entry = tk.Entry(self, width=62)
        self.model_entry.insert(0, default_model_dir)
        self.model_entry.grid(row=2, column=1, padx=6, sticky="ew")
        tk.Button(self, text="选择", width=8, command=self.choose_model_path).grid(row=2, column=2, padx=6)

        tk.Label(self, text="音/视频文件:").grid(row=3, column=0, sticky="w", padx=12, pady=8)
        self.media_entry = tk.Entry(self, width=62)
        self.media_entry.grid(row=3, column=1, padx=6, sticky="ew")
        tk.Button(self, text="选择", width=8, command=self.choose_media_file).grid(row=3, column=2, padx=6)

        tk.Label(self, text="输出格式:").grid(row=4, column=0, sticky="w", padx=12, pady=8)
        self.output_fmt = tk.StringVar(value="srt")
        tk.Radiobutton(self, text="SRT", variable=self.output_fmt, value="srt").grid(row=4, column=1, sticky="w")
        tk.Radiobutton(self, text="TXT", variable=self.output_fmt, value="txt").grid(row=4, column=1)
        self.use_context_var = tk.BooleanVar(value=use_context)
        tk.Checkbutton(self, text="使用上下文", variable=self.use_context_var).grid(row=4, column=2, sticky="w", padx=6)
        self.beam_search_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Beam Search", variable=self.beam_search_var).grid(row=4, column=3, sticky="w", padx=6)

        tk.Label(self, text="识别语言:").grid(row=5, column=0, sticky="w", padx=12)
        self.lang_combo = ttk.Combobox(self, values=["zh", "auto"], width=8, state="readonly")
        self.lang_combo.set("zh")
        self.lang_combo.grid(row=5, column=1, sticky="w", padx=6)

        tk.Label(self, text="进度:").grid(row=6, column=0, sticky="w", padx=12, pady=8)
        self.progress = ttk.Progressbar(self, orient="horizontal", length=560, mode="determinate", maximum=100)
        self.progress.grid(row=6, column=1, columnspan=3, padx=6, sticky="ew")
        self.progress_pct = tk.Label(self, text="0%")
        self.progress_pct.grid(row=6, column=3, sticky="e", padx=10)

        self.start_button = tk.Button(self, text="开始转写", width=12, command=self.on_run)
        self.start_button.grid(row=7, column=1, pady=6, sticky="e")
        self.stop_button = tk.Button(self, text="停止", width=12, command=self.on_stop, state="disabled")
        self.stop_button.grid(row=7, column=0, pady=6, padx=12, sticky="w")
        tk.Button(self, text="打开输出所在文件夹", width=18, command=self.on_open_folder).grid(row=7, column=2, pady=6, sticky="w")

        tk.Label(self, text="日志:").grid(row=8, column=0, sticky="nw", padx=12, pady=8)
        self.log_text = tk.Text(self, height=10, width=92, state="disabled")
        self.log_text.grid(row=8, column=1, columnspan=3, padx=6, pady=6, sticky="nsew")
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

    def on_device_change(self, event=None):
        device = self.device_var.get()
        if device == "cpu":
            vals = ["int8", "int16", "int32"]
            self.compute_type_combo["values"] = vals
            self.compute_type_combo.config(state="readonly")
            self.compute_type_combo.current(0)
            self.compute_type_var.set(vals[0])
        elif device == "gpu":
            vals = ["float16", "float32"]
            self.compute_type_combo["values"] = vals
            self.compute_type_combo.config(state="readonly")
            self.compute_type_combo.current(0)
            self.compute_type_var.set(vals[0])
        else:
            self.compute_type_var.set("")
            self.compute_type_combo["values"] = []
            self.compute_type_combo.config(state="disabled")

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
                status, payload = msg[1], msg[2]
                self.progress.stop()
                if status is True:
                    self.last_output = payload
                    self.log(f"✅ 完成：{payload}")
                    self.progress["value"] = 100
                    self.progress_pct.config(text="100%")
                    messagebox.showinfo("完成", f"输出文件：{payload}")
                elif status == "CANCEL":
                    self.log("⛔ 已取消")
                    self.progress["value"] = 0
                    self.progress_pct.config(text="0%")
                    messagebox.showinfo("已取消", payload)
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
        self.stop_button.config(state="normal" if running else "disabled")

    def on_open_folder(self):
        if self.last_output and os.path.exists(self.last_output):
            open_in_explorer(self.last_output)
        else:
            messagebox.showinfo("提示", "还没有生成任何文件。")

    def on_stop(self):
        if getattr(self, "stop_event", None):
            self.stop_event.set()
            self.log("⏹️ 正在停止...")

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
        device_mode = self.device_var.get()
        self.log(f"设备：{device_mode}")
        compute_type = self.compute_type_var.get().strip()
        if compute_type:
            self.log(f"精度：{compute_type}")
        else:
            compute_type = None
        self.set_running(True)
        self.progress.configure(mode="determinate")
        self.progress["value"] = 0
        self.progress_pct.config(text="0%")

        def progress_cb(v):
            self.enqueue(("PROG", v))
        self.stop_event = threading.Event()

        def worker():
            self.enqueue("[DEBUG] worker thread started")
            try:
                outfile = transcribe_with_progress(
                    model_path=model_path,
                    media_path=media_path,
                    fmt=fmt,
                    language=(None if lang == "auto" else lang),
                    backend=backend,
                    logger=self.enqueue,
                    progress_cb=progress_cb,
                    stop_event=self.stop_event,
                    device_mode=device_mode,
                    compute_type=compute_type,
                    use_context=self.use_context_var.get(),
                    beam_search=self.beam_search_var.get(),
                    beam_width=DEFAULT_BEAM_WIDTH,
                    n_best=DEFAULT_N_BEST,
                )
                self.enqueue(("DONE", True, outfile))
            except TranscriptionStopped:
                self.enqueue(("DONE", "CANCEL", "任务已停止"))
            except Exception as e:
                err = f"{e}\n{traceback.format_exc()}"
                self.enqueue(("DONE", False, err))
            finally:
                self.enqueue("[DEBUG] worker thread finished")

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-context", action="store_true", help="启用跨块上下文提示")
    args = parser.parse_args()
    WhisperApp(use_context=args.use_context).mainloop()
