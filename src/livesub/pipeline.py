"""即時管線：擷取 → 切句 → 辨識 → 翻譯 → 畫面／逐字稿。

執行緒配置（彼此以佇列溝通，任何一段變慢都不會卡住音訊擷取）：

    audio-capture ──► segmenter ──► asr ──► translate ──► UI 訊號 / 逐字稿
                       (VAD)   (本機 GPU    (雲端 API)
                                或雲端 API)

即時性策略：
* 定稿（final）用有上限的佇列，塞車時丟掉最舊的並提示「辨識跟不上」。
* 中間結果（partial）用「只保留最新一筆」的信箱，永遠不會累積延遲。

名詞解釋：
* 定稿（final）：講者停頓、一句話講完後的最終結果，會寫入逐字稿。
* 中間結果（partial）：句子還沒講完時的暫時結果，只顯示在畫面上，之後會被定稿取代。
* 佇列（queue）：執行緒之間傳資料的「輸送帶」，一端放入、另一端取出。

為什麼要分成多個執行緒？
    辨識、翻譯都要呼叫網路 API，一次要等 0.5~2 秒。如果全部在同一個執行緒裡，
    等待期間就收不到新的聲音。拆開後每段各做各的，音訊擷取永遠不會被卡住。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

from .asr import Transcriber
from .audio import AudioCapture, AudioChunk, UtteranceSegmenter, VoiceDetector, put_dropping_oldest
from .config import AppConfig
from .core.budget import BudgetLedger
from .core.http import CloudError
from .core.models import Stage, SubtitleEvent, Transcription, Utterance
from .output import TranscriptWriter
from .translation import ChineseNormalizer, PassthroughTranslator, TranslationError, Translator

log = logging.getLogger(__name__)

T = TypeVar("T")
EventCallback = Callable[[SubtitleEvent], None]
StatusCallback = Callable[[str], None]

_POLL_S = 0.05  # 各迴圈等待新資料的間隔（秒）；越短反應越快，但會多耗一點 CPU
_STREAM_UI_INTERVAL_S = 0.1  # 串流翻譯時，畫面最多每 0.1 秒更新一次


class LatestOnly(Generic[T]):
    """只保留最新一筆的信箱（執行緒安全）。

    用於中間結果：辨識還沒做完時又來了新的快照，舊的已經沒有意義，直接覆蓋掉。
    這樣就算辨識比較慢，也只會處理「最新」的內容，延遲不會越積越多。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None

    def put(self, item: T) -> None:
        with self._lock:
            self._item = item

    def take(self) -> T | None:
        with self._lock:
            item, self._item = self._item, None
            return item


class PartialThrottle:
    """控制中間結果翻譯的頻率，避免浪費 API 費用。

    同一句話的中間結果會一直更新，若每次都送去翻譯會很貴。
    只有「距離上次翻譯夠久」而且「多出夠多新字」時才再翻一次。
    """

    def __init__(self, min_interval_s: float, min_new_chars: int) -> None:
        self._interval = min_interval_s
        self._min_chars = min_new_chars
        # 每一句（seq）上次翻譯的時間與當時的文字長度
        self._state: dict[int, tuple[float, int]] = {}

    def should_translate(self, seq: int, text: str, now: float) -> bool:
        """這次的中間結果值得翻譯嗎？值得的話順便記下這次的時間與長度。"""
        last = self._state.get(seq)
        if last is not None:
            last_t, last_len = last
            if now - last_t < self._interval or len(text) - last_len < self._min_chars:
                return False
        self._state[seq] = (now, len(text))
        return True

    def forget(self, seq: int) -> None:
        """這句已經定稿，不再需要追蹤。"""
        self._state.pop(seq, None)
        if len(self._state) > 64:  # 保險：避免異常情況下紀錄無限增加
            self._state.clear()


class Pipeline:
    """組裝並管理所有背景執行緒。

    app.py 建立好辨識器、翻譯器等元件後交給 Pipeline，
    Pipeline 負責把它們用佇列串起來，並在背景執行。
    結果透過 on_event（字幕）與 on_status（狀態文字）回呼通知畫面。
    """

    def __init__(
        self,
        cfg: AppConfig,
        transcriber: Transcriber,
        translator: Translator,
        vad_factory: Callable[[], VoiceDetector],
        writer: TranscriptWriter,
        ledger: BudgetLedger | None,
        on_event: EventCallback,
        on_status: StatusCallback,
    ) -> None:
        self.cfg = cfg
        self.transcriber = transcriber
        self.translator = translator
        self.ledger = ledger
        self._vad_factory = vad_factory
        self._writer = writer
        self._on_event = on_event
        self._on_status = on_status

        self._t0 = time.monotonic()  # 工作階段開始時間；字幕時間軸都相對於它
        self._stop = threading.Event()  # 設定後所有迴圈結束
        self._paused = threading.Event()  # 設定後暫停處理（不花錢）
        # 各段之間的「輸送帶」：
        #   擷取 → _audio_q → 切句 → _final_utts / _partial_utt → 辨識
        #   辨識 → _final_trs / _partial_tr → 翻譯
        self._audio_q: queue.Queue[AudioChunk] = queue.Queue(maxsize=cfg.audio.max_queue_chunks)
        self._final_utts: queue.Queue[Utterance] = queue.Queue(maxsize=cfg.asr.max_backlog)
        self._partial_utt: LatestOnly[Utterance] = LatestOnly()
        self._final_trs: queue.Queue[Transcription] = queue.Queue(maxsize=100)
        self._partial_tr: LatestOnly[Transcription] = LatestOnly()
        self._finalized_seq = 0  # 已定稿的最大序號（ASR 執行緒寫入）
        self._throttle = PartialThrottle(
            cfg.translate.partial_min_interval_s, cfg.translate.partial_min_new_chars
        )
        self._threads: list[threading.Thread] = []
        self._last_status: tuple[str, float] = ("", 0.0)
        self.capture = AudioCapture(cfg.audio, self._audio_q, self.clock, self._status)
        self._stopped = False
        # 中文語音不需翻譯，只做繁體／台灣用語轉換
        self._zh = PassthroughTranslator(ChineseNormalizer(cfg.translate.taiwan_convert))

    # ------------------------------------------------------------------ 控制
    def clock(self) -> float:
        """工作階段開始後經過的秒數。"""
        return time.monotonic() - self._t0

    def start(self) -> None:
        """在背景啟動（載入模型可能要很久，不能卡住畫面）。"""
        threading.Thread(target=self._bootstrap, name="bootstrap", daemon=True).start()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, paused: bool) -> None:
        (self._paused.set if paused else self._paused.clear)()
        self._status("已暫停" if paused else f"聆聽中：{self.capture.device_name}")

    def stop(self, timeout: float = 3.0) -> None:
        """停止所有執行緒並關閉檔案。可重複呼叫。"""
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        self.capture.stop(timeout)
        for t in self._threads:
            t.join(timeout)
        self._writer.close()
        if self.ledger:
            self.ledger.save()
        log.info("管線已停止")

    # ------------------------------------------------------------------ 啟動
    def _bootstrap(self) -> None:
        """依序：載入辨識器 → 預熱翻譯器 → 啟動三個處理執行緒 → 最後才開始收音。

        先開處理執行緒再收音，確保收進來的聲音一定有人處理。
        """
        try:
            self._status("準備語音辨識中…" if self.transcriber.is_cloud
                         else "載入語音辨識模型中（第一次會下載，請稍候）…")
            self.transcriber.load()
            self.translator.warmup()
        except Exception as exc:  # noqa: BLE001 - 顯示給使用者
            log.exception("啟動失敗")
            self._status(f"啟動失敗：{exc}")
            return
        # 是否產生中間結果：on = 一定要；auto = 只有辨識器夠快（本機 GPU）才要
        mode = self.cfg.asr.partial_results
        emit_partials = mode == "on" or (mode == "auto" and self.transcriber.realtime_partials)
        if mode == "auto" and not emit_partials:
            log.info("辨識引擎 %s 不適合即時中間結果，只輸出定稿", self.transcriber.name)
        segmenter = UtteranceSegmenter(self.cfg.segmenter, self._vad_factory(), emit_partials=emit_partials)
        for name, target in (
            ("segmenter", lambda: self._segment_loop(segmenter)),
            ("asr", self._asr_loop),
            ("translate", self._translate_loop),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        if not self._stop.is_set():
            self.capture.start()

    def _status(self, message: str, dedupe_s: float = 0.0) -> None:
        """更新畫面上的狀態文字。dedupe_s 秒內重複的相同訊息會略過，避免洗版。"""
        last_msg, last_t = self._last_status
        now = time.monotonic()
        if dedupe_s and message == last_msg and now - last_t < dedupe_s:
            return
        self._last_status = (message, now)
        self._on_status(message)

    # ------------------------------------------------------------------ 切句
    def _segment_loop(self, seg: UtteranceSegmenter) -> None:
        """切句執行緒：從音訊佇列取 50 毫秒的小塊，交給切句器判斷句子的開頭與結尾。"""
        was_paused = False
        while not self._stop.is_set():
            try:
                samples, t = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                # 一段時間沒有音訊（例如電腦沒播聲音）：讓切句器有機會把講到一半的句子收尾
                self._dispatch(seg.on_idle(self.clock()))
                continue
            if self._paused.is_set():
                if not was_paused:  # 剛進入暫停：丟掉講到一半的句子
                    seg.reset()
                    was_paused = True
                continue
            was_paused = False
            self._dispatch(seg.push(samples, t))

    def _dispatch(self, utts: list[Utterance]) -> None:
        """把切好的句子送往辨識：中間快照放「最新信箱」，定稿放佇列（滿了丟最舊的）。"""
        for u in utts:
            if not u.is_final:
                self._partial_utt.put(u)
            elif put_dropping_oldest(self._final_utts, u):
                log.warning("辨識速度跟不上，已略過較舊的句子")
                self._status("辨識速度跟不上，已略過一句（可改用較小模型）", dedupe_s=10)

    # ------------------------------------------------------------------ 辨識
    def _next_utterance(self) -> Utterance | None:
        """取下一個要辨識的音訊：定稿優先；沒有定稿才處理中間快照。

        已經定稿的句子，它的舊快照就不用再辨識了（partial.seq > _finalized_seq 的判斷）。
        """
        try:
            return self._final_utts.get(timeout=_POLL_S)
        except queue.Empty:
            pass
        partial = self._partial_utt.take()
        if partial is not None and partial.seq > self._finalized_seq:
            return partial
        return None

    def _asr_loop(self) -> None:
        """辨識執行緒：音訊 → 文字。定稿交給翻譯執行緒；中間結果先直接顯示原文。"""
        while not self._stop.is_set():
            utt = self._next_utterance()
            if utt is None:
                continue
            tr = self._transcribe(utt)
            if utt.is_final:
                self._finalized_seq = max(self._finalized_seq, utt.seq)
            if tr is None:
                if utt.is_final:  # 讓畫面移除這句的中間結果
                    self._final_trs.put(self._empty(utt))
                continue
            if tr.is_final:
                if self._needs_translation(tr):
                    # 先把原文顯示出來，不必等翻譯完成（翻譯通常還要 1 秒左右）
                    self._on_event(self._event(tr, Stage.PARTIAL_SOURCE))
                self._final_trs.put(tr)
            elif tr.seq > self._finalized_seq:
                self._on_event(self._event(tr, Stage.PARTIAL_SOURCE))
                self._partial_tr.put(tr)

    def _transcribe(self, utt: Utterance) -> Transcription | None:
        """辨識一段音訊；預算用完或失敗時回傳 None（單句失敗不影響後面的句子）。"""
        if not self._cloud_allowed(self.transcriber.is_cloud, partial=not utt.is_final):
            if utt.is_final:
                self._status("本月雲端預算已用完，暫停語音辨識", dedupe_s=300)
            return None
        try:
            return self.transcriber.transcribe(utt)
        except CloudError as exc:
            log.warning("辨識失敗 seq=%d：%s", utt.seq, exc)
            if utt.is_final:
                self._status(f"語音辨識失敗：{exc}", dedupe_s=30)
        except Exception:  # noqa: BLE001 - 單句失敗不應中止整個管線
            log.exception("辨識失敗 seq=%d", utt.seq)
        return None

    @staticmethod
    def _empty(utt: Utterance) -> Transcription:
        """空白的定稿：辨識失敗時用來通知畫面「這句結束了」，把殘留的中間結果清掉。"""
        return Transcription(utt.seq, True, utt.started_at, utt.ended_at, "", 0.0, "", 0.0)

    @staticmethod
    def _event(tr: Transcription, stage: Stage, **kw: object) -> SubtitleEvent:
        return SubtitleEvent(
            seq=tr.seq,
            stage=stage,
            started_at=tr.started_at,
            ended_at=tr.ended_at,
            language=tr.language,
            source_text=tr.text,
            asr_latency_s=tr.asr_latency_s,
            **kw,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ 翻譯
    def _translate_loop(self) -> None:
        """翻譯執行緒：定稿優先翻譯；空檔時才翻譯最新的中間結果。"""
        while not self._stop.is_set():
            try:
                tr = self._final_trs.get(timeout=_POLL_S)
            except queue.Empty:
                partial = self._partial_tr.take()
                if partial is not None and partial.seq > self._finalized_seq:
                    self._translate_partial(partial)
                continue
            self._translate_final(tr)

    def _needs_translation(self, tr: Transcription) -> bool:
        """這句會送去翻譯嗎？（中文只做繁體轉換、選了「不翻譯」時都不用）"""
        return bool(tr.text) and tr.language != "zh" and self.translator.name != "none"

    def _cloud_allowed(self, is_cloud: bool, partial: bool) -> bool:
        """預算檢查：本機服務永遠允許；雲端服務交給費用帳本判斷（中間結果會比定稿先被停掉）。"""
        if not is_cloud or self.ledger is None:
            return True
        return self.ledger.allows(partial)

    def _translate_partial(self, tr: Transcription) -> None:
        """翻譯中間結果（失敗就算了，反正定稿還會再翻一次）。"""
        cfg = self.cfg.translate
        if not cfg.partial_translation or tr.language == "zh" or self.translator.name == "none":
            return
        if not self._throttle.should_translate(tr.seq, tr.text, time.monotonic()):
            return
        if not self._cloud_allowed(self.translator.is_cloud, partial=True):
            return
        try:
            res = self.translator.translate(tr.text, tr.language, partial=True)
        except TranslationError as exc:
            log.debug("中間結果翻譯失敗：%s", exc)
            return
        except Exception:  # noqa: BLE001 - 非預期錯誤不能讓翻譯執行緒整個停掉
            log.exception("中間結果翻譯發生非預期錯誤 seq=%d", tr.seq)
            return
        # 翻譯期間這句可能已經定稿了，那就不要再顯示過時的中間譯文
        if tr.seq > self._finalized_seq and res.text:
            self._on_event(
                self._event(tr, Stage.PARTIAL_TRANSLATION, translated_text=res.text,
                            translate_latency_s=res.latency_s)
            )

    def _stream_updater(self, tr: Transcription) -> Callable[[str], None]:
        """回傳給串流翻譯用的函式：收到目前為止的譯文就更新畫面（最多每 0.1 秒一次，避免閃爍）。"""
        last = [0.0]

        def update(partial_text: str) -> None:
            now = time.monotonic()
            if partial_text and now - last[0] >= _STREAM_UI_INTERVAL_S:
                last[0] = now
                self._on_event(self._event(tr, Stage.PARTIAL_TRANSLATION, translated_text=partial_text))

        return update

    def _translate_final(self, tr: Transcription) -> None:
        """翻譯定稿，寫入逐字稿並顯示。中文語音不送翻譯，只做繁體轉換。"""
        self._throttle.forget(tr.seq)
        if not tr.text:  # 辨識不出內容（或失敗）：只通知畫面清掉這句
            self._on_event(self._event(tr, Stage.FINAL))
            return
        translated, latency, error = "", 0.0, None
        if tr.language == "zh":
            translated = self._zh.translate(tr.text, "zh").text
        elif self.translator.name != "none":
            if self._cloud_allowed(self.translator.is_cloud, partial=False):
                try:
                    if getattr(self.translator, "supports_streaming", False):
                        # 串流：譯文一邊產生一邊顯示，不用等整句翻完
                        res = self.translator.translate(tr.text, tr.language,
                                                        on_delta=self._stream_updater(tr))
                    else:
                        res = self.translator.translate(tr.text, tr.language)
                    translated, latency = res.text, res.latency_s
                except TranslationError as exc:
                    error = str(exc)
                    log.warning("翻譯失敗 seq=%d：%s", tr.seq, exc)
                    self._status(f"翻譯失敗：{exc}", dedupe_s=30)
                except Exception as exc:  # noqa: BLE001 - 非預期錯誤：這句只顯示原文，執行緒繼續運作
                    error = f"非預期錯誤：{exc}"
                    log.exception("翻譯發生非預期錯誤 seq=%d", tr.seq)
                    self._status(f"翻譯失敗：{exc}", dedupe_s=30)
            else:
                error = "本月雲端預算已用完"
                self._status("本月雲端預算已用完，僅顯示原文", dedupe_s=300)
        evt = self._event(tr, Stage.FINAL, translated_text=translated,
                          translate_latency_s=latency, error=error)
        self._writer.write(evt)
        self._on_event(evt)
        log.debug(
            "seq=%d lang=%s asr=%.2fs tr=%.2fs | %s → %s",
            tr.seq, tr.language, tr.asr_latency_s, latency, tr.text, translated,
        )
