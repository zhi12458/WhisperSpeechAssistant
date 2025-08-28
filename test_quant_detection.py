import os
import json
import stat
import tempfile

# create a fake ffmpeg so importing whisper_assistant succeeds
_ffmpeg_tmp = tempfile.mkdtemp()
_ffmpeg_stub = os.path.join(_ffmpeg_tmp, 'ffmpeg')
with open(_ffmpeg_stub, 'w', encoding='utf-8') as f:
    f.write('#!/bin/sh\nexit 0\n')
os.chmod(_ffmpeg_stub, stat.S_IRWXU)
os.environ['PATH'] = _ffmpeg_tmp + os.pathsep + os.environ.get('PATH', '')

from whisper_assistant import detect_model_device_compute


def _make_model_dir(name: str, quant: str | None = None) -> str:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, name)
    os.makedirs(path)
    if quant:
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({'quantization': quant}, f)
    return path


# directory suffixes only
assert detect_model_device_compute(_make_model_dir('foo-ct2int16')) == ("cpu", "int16")
assert detect_model_device_compute(_make_model_dir('foo-i8f16')) == ("cpu", "int8")
assert detect_model_device_compute(_make_model_dir('foo-int8')) == ("cpu", "int8")

# config.json only
assert detect_model_device_compute(_make_model_dir('foo', 'i16')) == ("cpu", "int16")

# directory name should override conflicting config.json
assert detect_model_device_compute(_make_model_dir('foo-ct2int16', 'int8')) == ("cpu", "int16")

print('quantization detection tests passed')
