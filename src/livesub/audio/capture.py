"""Windows 系統音訊擷取（WASAPI loopback），轉為 16 kHz 單聲道 float32。

使用 PyAudioWPatch（PortAudio 的 Windows loopback 分支）。
注意：PortAudio 在初始化時才會讀取裝置清單，因此切換耳機／喇叭後，
需要從系統匣選單按「重新連接音訊裝置」。

名詞解釋：
* WASAPI loopback：Windows 提供的功能，可以「錄下喇叭正在播放的聲音」，
  所以不需要虛擬音效線就能收到會議軟體、影片的聲音。
* 取樣率：每秒幾個樣本。音效卡通常是 48000 Hz，語音辨識只需要 16000 Hz，所以要重新取樣。
* 聲道：喇叭通常是立體聲（2 聲道），語音辨識只需要單聲道，所以把左右聲道平均。

三種來源（audio.source）：
* loopback：只收電腦播放的聲音
* mic：只收麥克風
* both：同時開兩個裝置，用 Mixer 混成一路
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
    """int16 交錯（interleaved）PCM 轉為單聲道 float32（-1~1）。

    交錯格式：立體聲資料排成 左,右,左,右…；轉成 (N, 2) 的表格後每列取平均就是單聲道。
    """
    # int16 範圍是 -32768~32767，除以 32768 變成 -1~1 的浮點數
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable = pcm.size - pcm.size % channels  # 保險：丟掉不完整的最後一組
        pcm = pcm[:usable].reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32, copy=False)


class Resampler:
    """串流式重新取樣。優先使用 soxr（高品質、有狀態），否則線性內插。

    「串流式」：音訊是一小塊一小塊進來的，soxr 會記住上一塊的尾巴，
    讓塊與塊的接縫處不會產生雜音。
    """

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
        """把一塊音訊從輸入取樣率轉成輸出取樣率。"""
        if self._in == self._out or x.size == 0:
            return x
        if self._stream is not None:
            return np.asarray(self._stream.resample_chunk(x), dtype=np.float32)
        # 備援：線性內插（在原本的樣本之間畫直線取點）
        n_out = int(round(x.size * self._out / self._in))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.interp(np.linspace(0, x.size - 1, n_out), np.arange(x.size), x).astype(np.float32)


def put_dropping_oldest(q: queue.Queue[Any], item: Any) -> bool:
    """放入佇列；滿了就丟掉最舊的一筆。回傳是否有丟棄。

    即時字幕寧可丟掉舊資料，也不要讓延遲越積越多。
    """
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


class Mixer:
    """把兩路 16 kHz 音訊混成一路（source = "both"）。

    麥克風是主時鐘：每收到一塊麥克風音訊，就取出等長的系統聲音一起相加。
    WASAPI loopback 在電腦沒有播放聲音時可能完全不送資料，所以不能等它；
    不足的部分補 0。系統聲音緩衝最多保留 max_lag_s，超過就丟掉最舊的，避免延遲越積越多。
    """

    def __init__(self, sample_rate: int = 16_000, max_lag_s: float = 0.5) -> None:
        self._max = max(1, int(sample_rate * max_lag_s))
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def push_secondary(self, x: AudioArray) -> None:
        if x.size == 0:
            return
        with self._lock:
            self._buf = np.concatenate([self._buf, x])
            if self._buf.size > self._max:
                self._buf = self._buf[-self._max:]

    def mix_primary(self, x: AudioArray) -> AudioArray:
        with self._lock:
            n = min(x.size, self._buf.size)
            other, self._buf = self._buf[:n], self._buf[n:]
        out = x.astype(np.float32, copy=True)
        out[:n] += other
        np.clip(out, -1.0, 1.0, out=out)
        return out


class AudioCapture:
    """在背景執行緒擷取音訊並放入佇列。

    音效卡每 50 毫秒呼叫一次 callback，把資料轉成 16 kHz 單聲道後放進 out_queue，
    由 pipeline 的切句執行緒取出。裝置出錯時會自動重試（等待時間 1、2、4… 最多 15 秒）。
    """

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
        """執行緒主體：開啟裝置 → 擷取 → 出錯或要求重連時關閉，再重新開始。"""
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
                self._capture(pa, pyaudio, self._resolve_devices(pa))
                backoff = 1.0
            except (OSError, LookupError) as exc:
                log.warning("音訊裝置錯誤：%s", exc)
                self._status(f"音訊裝置錯誤，{backoff:.0f} 秒後重試：{exc}")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
            finally:
                pa.terminate()
            self._reconnect.clear()

    @staticmethod
    def _loopback_info(pa: Any, index: int) -> dict[str, Any]:
        if index >= 0:
            return dict(pa.get_device_info_by_index(index))
        return dict(pa.get_default_wasapi_loopback())

    @staticmethod
    def _mic_info(pa: Any, index: int) -> dict[str, Any]:
        if index >= 0:
            return dict(pa.get_device_info_by_index(index))
        return dict(pa.get_default_input_device_info())

    def _resolve_devices(self, pa: Any) -> list[dict[str, Any]]:
        """回傳要開啟的裝置；both 時第一個是麥克風（主時鐘），第二個是系統聲音。"""
        cfg = self._cfg
        if cfg.source == "loopback":
            return [self._loopback_info(pa, cfg.device_index)]
        if cfg.source == "mic":
            return [self._mic_info(pa, cfg.device_index)]
        return [self._mic_info(pa, cfg.mic_device_index), self._loopback_info(pa, cfg.device_index)]

    def _emit(self, mono: AudioArray) -> None:
        """把處理好的音訊（附上目前時間）放進佇列給切句器。"""
        if mono.size and put_dropping_oldest(self._q, (mono, self._clock())):
            self.dropped_chunks += 1

    def _open(self, pa: Any, pyaudio: Any, info: dict[str, Any],
              sink: Callable[[AudioArray], None]) -> Any:
        """開啟一個裝置的音訊串流；每收到一塊資料就轉成 16 kHz 單聲道交給 sink。"""
        channels = max(1, int(info["maxInputChannels"]))
        rate = int(info["defaultSampleRate"])
        resampler = Resampler(rate, self._cfg.target_sample_rate)

        # callback 由音效驅動程式的執行緒呼叫，必須很快返回，而且不能丟出例外
        def callback(in_data: bytes, frame_count: int, _time: Any, _status: int) -> tuple[None, int]:
            try:
                sink(resampler.process(to_mono_float(in_data, channels)))
            except Exception:  # noqa: BLE001 - callback 中的例外不可外拋
                log.exception("音訊 callback 發生錯誤")
            return None, pyaudio.paContinue

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(info["index"]),
            frames_per_buffer=max(256, int(rate * 0.05)),  # 50 ms
            stream_callback=callback,
        )
        log.info("開始擷取：%s（%d Hz, %d ch）", info.get("name", "?"), rate, channels)
        return stream

    def _capture(self, pa: Any, pyaudio: Any, infos: list[dict[str, Any]]) -> None:
        """開啟所有裝置並持續擷取，直到停止、要求重連或裝置被移除。"""
        if len(infos) == 1:
            # 單一來源：收到就直接送出
            sinks: list[Callable[[AudioArray], None]] = [self._emit]
        else:
            # 雙來源：麥克風收到時混入系統聲音再送出；系統聲音只先暫存在 Mixer
            mixer = Mixer(self._cfg.target_sample_rate)
            sinks = [lambda x: self._emit(mixer.mix_primary(x)), mixer.push_secondary]
        self.device_name = " + ".join(str(i.get("name", "?")) for i in infos)

        streams: list[Any] = []
        try:
            for info, sink in zip(infos, sinks, strict=True):
                streams.append(self._open(pa, pyaudio, info, sink))
            self._status(f"聆聽中：{self.device_name}")
            for s in streams:
                s.start_stream()
            # 資料由 callback 在背景處理；這裡只每 0.5 秒檢查一次裝置是否還活著
            while not self._stop.is_set() and not self._reconnect.is_set():
                self._reconnect.wait(0.5)
                if not all(s.is_active() for s in streams):
                    raise OSError("音訊串流已停止（裝置可能被移除）")
        finally:
            for s in streams:
                try:
                    s.stop_stream()
                    s.close()
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
