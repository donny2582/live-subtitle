"""音訊模組：擷取（WASAPI loopback）＋ VAD 切句。"""

from .capture import AudioCapture, AudioChunk, list_devices, put_dropping_oldest
from .segmenter import EnergyVad, UtteranceSegmenter, VoiceDetector, make_vad

__all__ = [
    "AudioCapture",
    "AudioChunk",
    "EnergyVad",
    "UtteranceSegmenter",
    "VoiceDetector",
    "list_devices",
    "make_vad",
    "put_dropping_oldest",
]
