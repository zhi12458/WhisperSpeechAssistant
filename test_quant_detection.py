import os
import stat
import tempfile
import types
import sys

# create a fake ffmpeg so importing whisper_assistant succeeds
_ffmpeg_tmp = tempfile.mkdtemp()
_ffmpeg_stub = os.path.join(_ffmpeg_tmp, 'ffmpeg')
with open(_ffmpeg_stub, 'w', encoding='utf-8') as f:
    f.write('#!/bin/sh\nexit 0\n')
os.chmod(_ffmpeg_stub, stat.S_IRWXU)
os.environ['PATH'] = _ffmpeg_tmp + os.pathsep + os.environ.get('PATH', '')

from whisper_assistant import detect_model_device_compute, load_model


def _make_model_dir(name: str) -> str:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, name)
    os.makedirs(path)
    return path


assert detect_model_device_compute(_make_model_dir('belle-whisper-large-v3-turbo-ct2int16')) == ("cpu", "int16")
assert detect_model_device_compute(_make_model_dir('belle-whisper-large-v3-turbo-ct2i8')) == ("cpu", "int8")
assert detect_model_device_compute(_make_model_dir('belle-whisper-large-v3-turbo-ct2i8f16')) == ("cpu", "int8")
assert detect_model_device_compute(_make_model_dir('belle-whisper-large-v3-turbo-ct2f16')) is None

# simulate GPU for float16/int8_float16 detection
fake_c2 = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
sys.modules['ctranslate2'] = fake_c2
try:
    assert detect_model_device_compute(_make_model_dir('belle-whisper-large-v3-turbo-ct2f16')) == ("cuda", "float16")
    assert detect_model_device_compute(_make_model_dir('belle-whisper-large-v3-turbo-ct2i8f16')) == ("cuda", "float16")
finally:
    sys.modules.pop('ctranslate2', None)

# load_model should not fall back to int8 when int16 fails
import whisper_assistant
orig_cls = whisper_assistant.WhisperModel
class FakeWhisperModel:
    def __init__(self, model_path, device, compute_type):
        if compute_type == 'int8':
            self.compute_type = compute_type
        else:
            raise RuntimeError('unsupported')
whisper_assistant.WhisperModel = FakeWhisperModel
try:
    try:
        load_model(_make_model_dir('belle-whisper-large-v3-turbo-ct2int16'), 'ct2')
        raise AssertionError('expected failure')
    except RuntimeError:
        pass
finally:
    whisper_assistant.WhisperModel = orig_cls

print('quantization detection tests passed')
