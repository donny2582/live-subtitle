"""Windows 系統音訊擷取（WASAPI loopback），轉為 16 kHz 單聲道 float32。

使用 PyAudioWPatch（PortAudio 的 Windows loopback 分支）。
注意：PortAudio 在初始化時才會讀取裝置清單，因此切換耳機／喇叭後，
需要從系統匣選單按「重新連接音訊裝置」。
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from ..config import AudioConfig
from ..core.models import AudioArray

log = logging.getLogger(__name__)

AudioChunk = tuple[AudioArray, float]  # (16 kHz 樣本, 結束時間)


def to_mono_float(data: bytes, channels: int) -> AudioArray:
    """int16 交錯（interleaved）PCM 轉為單聲道 float32（-1~1）。"""
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable = pcm.size - pcm.size % channels
        pcm = pcm[:usable].reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32, copy=False)


class Resampler:
    """串流式重新取樣。優先使用 soxr（高品質、有狀態），否則線性內插。"""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self._in, self._out = in_rate, out_rate
        self._stream: Any = None
        if in_rate == out_rate:
            return
        try:
            import soxr

            self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32", quality="HQ")
        except ImportError:
            log.warning("找不到 soxr，改用線性內插重新取樣（音質略差）")

    def process(self, x: AudioArray) -> AudioArray:
        if self._in == self._out or x.size == 0:
            return x
        if self._stream is not None:
            return np.asarray(self._stream.resample_chunk(x), dtype=np.float32)
        n_out = int(round(x.size * self._out / self._in))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.interp(np.linspace(0, x.size - 1, n_out), np.arange(x.size), x).astype(np.float32)


def put_dropping_oldest(q: queue.Queue[Any], item: Any) -> bool:
    """放入佇列；滿了就丟掉最舊的一筆。回傳是否有丟棄。"""
    dropped = False
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                pass


class AudioCapture:
    """在背景執行緒擷取音訊並放入佇列。"""

    def __init__(
        self,
        cfg: AudioConfig,
        out_queue: queue.Queue[AudioChunk],
        clock: Callable[[], float],
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._q = out_queue
        self._clock = clock
        self._status = on_status or (lambda _msg: None)
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._thread: threading.Thread | None = None
        self.device_name = ""
        self.dropped_chunks = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._reconnect.set()
        if self._thread:
            self._thread.join(timeout)

    def reconnect(self) -> None:
        """重新初始化 PortAudio 並重新選擇預設裝置。"""
        self._reconnect.set()

    # ------------------------------------------------------------------ 內部
    def _run(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            self._status("缺少 PyAudioWPatch（僅支援 Windows）")
            log.error("PyAudioWPatch 未安裝，無法擷取音訊")
            return
        backoff = 1.0
        while not self._stop.is_set():
            pa = pyaudio.PyAudio()
            try:
                info = self._resolve_device(pa)
                self._capture(pa, pyaudio, info)
                backoff = 1.0
            except (OSError, LookupError) as exc:
                log.warning("音訊裝置錯誤：%s", exc)
                self._status(f"音訊裝置錯誤，{backoff:.0f} 秒後重試：{exc}")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
            finally:
                pa.terminate()
            self._reconnect.clear()

    def _resolve_device(self, pa: Any) -> dict[str, Any]:
        if self._cfg.device_index >= 0:
            return dict(pa.get_device_info_by_index(self._cfg.device_index))
        if self._cfg.source == "loopback":
            return dict(pa.get_default_wasapi_loopback())
        return dict(pa.get_default_input_device_info())

    def _capture(self, pa: Any, pyaudio: Any, info: dict[str, Any]) -> None:
        channels = max(1, int(info["maxInputChannels"]))
        rate = int(info["defaultSampleRate"])
        frames = max(256, int(rate * 0.05))  # 50 ms
        resampler = Resampler(rate, self._cfg.target_sample_rate)
        self.device_name = str(info.get("name", "?"))

        def callback(in_data: bytes, frame_count: int, _time: Any, _status: int) -> tuple[None, int]:
            try:
                mono = resampler.process(to_mono_float(in_data, channels))
                if mono.size and put_dropping_oldest(self._q, (mono, self._clock())):
                    self.dropped_chunks += 1
            except Exception:  # noqa: BLE001 - callback 中的例外不可外拋
                log.exception("音訊 callback 發生錯誤")
            return None, pyaudio.paContinue

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(info["index"]),
            frames_per_buffer=frames,
            stream_callback=callback,
        )
        log.info("開始擷取：%s（%d Hz, %d ch）", self.device_name, rate, channels)
        self._status(f"聆聽中：{self.device_name}")
        try:
            stream.start_stream()
            while not self._stop.is_set() and not self._reconnect.is_set():
                self._reconnect.wait(0.5)
                if not stream.is_active():
                    raise OSError("音訊串流已停止（裝置可能被移除）")
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except OSError:
                pass


def list_devices() -> list[dict[str, Any]]:
    """列出可用的輸入與 loopback 裝置。"""
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        devices = []
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if int(d["maxInputChannels"]) <= 0:
                continue
            host = pa.get_host_api_info_by_index(d["hostApi"])["name"]
            devices.append(
                {
                    "index": i,
                    "name": d["name"],
                    "host_api": host,
                    "channels": int(d["maxInputChannels"]),
                    "rate": int(d["defaultSampleRate"]),
                    "loopback": bool(d.get("isLoopbackDevice", False)),
                }
            )
        return devices
    finally:
        pa.terminate()
