# LiveSub：Windows 即時語音翻譯字幕

把電腦正在播放的聲音（Teams／Zoom／Google Meet 會議、YouTube、本機影片）即時轉成
**英／日／韓 → 繁體中文（台灣用語）** 字幕，浮在所有視窗最上層，並自動存成逐字稿與 SRT 字幕檔。

| 功能 | 說明 |
|---|---|
| 擷取來源 | 系統播放聲音（WASAPI loopback），不需要虛擬音效線、不收您自己的麥克風 |
| 語音辨識 | **雲端**（OpenAI `gpt-4o-mini-transcribe`、Groq Whisper…，不需 GPU）或本機 faster-whisper，**逐句自動偵測語言** |
| 即時性 | 句子結束後約 1～2 秒出字幕；本機 GPU 模式另有每 0.7 秒的中間結果 |
| 翻譯 | OpenAI 相容 API（預設 `gpt-4.1-mini`）、**Claude**、DeepL；可換本機 Ollama |
| 台灣用語 | 提示詞要求台灣用語 + OpenCC `s2twp` 二次轉換 |
| 術語表 | `glossary.csv` 固定專有名詞譯法（已附交通安全術語範例） |
| 費用控管 | 辨識＋翻譯合計每月上限（預設 5 美元），快用完時先停中間結果，用完暫停雲端請求 |
| 字幕 | 雙行（原文小字＋中文大字）、可拖曳、Ctrl+滾輪調字級、點擊穿透 |
| 紀錄 | 每次啟動一個資料夾：`subtitles.srt`、`transcript.txt`、`transcript.jsonl` |

---

## 1. 系統需求

- Windows 10／11（64 位元）
- Python 3.11～3.14（建議 3.12）
- 雲端模式（預設）：不需要 GPU，只要網路與 API 金鑰
- 本機模式（選用）：建議 NVIDIA 顯示卡（VRAM 4 GB 以上）

## 2. 安裝（開發者模式）

所有套件都裝在專案內的 `.venv` 虛擬環境，不會影響系統的 Python。

```powershell
cd live-subtitle
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1             # 雲端辨識＋雲端翻譯（預設）
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -LocalAsr   # 另外安裝本機 faster-whisper
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Gpu        # 本機辨識＋NVIDIA CUDA DLL
```

| 檔案 | 內容 |
|---|---|
| `requirements.txt` | 核心（音訊、介面、雲端辨識與翻譯、anthropic SDK） |
| `requirements-local-asr.txt` | ＋ faster-whisper（`asr.engine = "local"` 才需要） |
| `requirements-gpu.txt` | ＋ CUDA 12／cuDNN 9 DLL |
| `requirements-dev.txt` | ＋ pytest、ruff、mypy、pyinstaller |

設定 API 金鑰（只要做一次，之後**新開的**終端機與程式都讀得到）：

```powershell
setx OPENAI_API_KEY "sk-..."          # 雲端語音辨識（預設）＋ OpenAI 翻譯
setx ANTHROPIC_API_KEY "sk-ant-..."   # translate.engine = "claude" 時
# Groq 辨識：setx GROQ_API_KEY "gsk_..."；DeepL：setx DEEPL_API_KEY "xxxx:fx"
```

> 金鑰刻意不寫進設定檔，避免不小心提交到 Git。

### 設定視窗（建議）

不想改檔案的話，用設定視窗就好：

- 程式執行中：**右鍵字幕**或**系統匣「字」圖示** →「設定…」
- 還沒啟動前：`scripts\run.ps1 -Settings`（只開設定視窗）

| 分頁 | 可以調整 |
|---|---|
| 服務與費用 | 語音辨識、翻譯用哪個服務（選了會自動填好模型與單價，並顯示 2 小時大約費用）、講到一半先翻譯、台灣用語、每月預算 |
| API 金鑰 | 貼上金鑰按「儲存」，存進 Windows 使用者環境變數（和 `setx` 一樣）；紅字代表目前選的服務需要但還沒設定 |
| 音訊與斷句 | 聲音來源與裝置、停頓多久算一句、雜音過濾、會出現的語言 |
| 字幕外觀 | 字型、字級、句數、寬度、透明度、位置（存檔後**立即套用**） |

- 滑鼠停在任何欄位上會出現說明。
- 除了字幕外觀，其他設定存檔後會詢問是否重新啟動套用。
- 只會改動你調整過的項目，`config.toml` 裡的其他內容與註解都會保留。
- 視窗裡沒有的細節，按「開啟設定檔（進階）」用記事本改，改完重新啟動。

### 切換服務（直接改設定檔）

`config.toml` 裡每個服務有自己的子區段（模型、金鑰環境變數、單價），切換只要改一行：

```toml
[asr]
engine = "cloud"        # cloud / local

[translate]
engine = "claude"       # openai / claude / deepl / none
```

## 3. 執行

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

- 本機模式第一次執行會下載 Whisper 模型（turbo 約 1.6 GB），字幕視窗會顯示「載入中」。
- **右鍵字幕**或**系統匣「字」圖示**可開啟選單：設定、暫停、點擊穿透、顯示原文、字級、開啟逐字稿資料夾。
- 左鍵點系統匣圖示可快速切換「點擊穿透」。
- 換耳機或喇叭後，請按選單中的「重新連接音訊裝置」。

其他指令：

```powershell
scripts\run.ps1 -Settings                    # 只開設定視窗
scripts\run.ps1 -ListDevices                 # 列出音訊裝置
scripts\run.ps1 --asr local --model small    # 臨時改用本機小模型
scripts\run.ps1 --engine claude              # 臨時改用 Claude 翻譯
scripts\run.ps1 --engine none                # 只要原文字幕，不翻譯
scripts\run.ps1 --log-level DEBUG            # 每句延遲寫進 logs\livesub.log
```

## 4. 打包成 .exe

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
# 需要看錯誤訊息時：scripts\build_exe.ps1 -Console
```

產出 `dist\LiveSub\LiveSub.exe`。整個 `LiveSub` 資料夾壓縮後即可帶到其他電腦，
`config.toml`、`glossary.csv`、`models\`、`transcripts\` 都會放在 exe 旁邊。

- 採用 one-dir 而不是單一 exe：含 CUDA DLL 時單一 exe 會有數 GB，每次啟動都要解壓到暫存資料夾，又慢又容易被防毒軟體誤判。
- 模型不打包，第一次執行時下載到 `models\`。要離線部署，可直接把開發機的 `models\` 資料夾複製過去。

## 5. 費用估算（請依實際用量與官網價格調整）

假設：一場 2 小時會議約 600 句、實際語音約 80 分鐘；每次翻譯約 600 輸入 token、50 輸出 token。

**語音辨識（雲端）**

| 設定 | 2 小時約 |
|---|---|
| OpenAI `gpt-4o-mini-transcribe`（預設，0.003 美元／分鐘） | 0.25 美元 |
| OpenAI `gpt-4o-transcribe` 或 `whisper-1`（0.006 美元／分鐘） | 0.5 美元 |
| Groq `whisper-large-v3-turbo`（每次最少計 10 秒） | 0.1 美元以下 |
| 本機 faster-whisper | 0 元（需 GPU） |

**翻譯**

| 設定 | 只翻定稿 | ＋中間結果翻譯 |
|---|---|---|
| `gpt-4.1-mini`（預設） | 0.2 美元 | 0.8 美元 |
| `gpt-4o-mini` | 0.07 美元 | 0.3 美元 |
| Claude `claude-haiku-5-5`，`thinking = "off"` | 0.05 美元 | 0.2 美元 |
| Claude `claude-opus-5-5`（思考無法關閉） | 3 美元以上 | 12 美元以上 |
| DeepL Free | 0 元 | 每月 50 萬字元 |

- 系統匣選單會顯示「本月雲端花費」（含辨識費用），累計存在 `usage_ledger.json`（跨月自動重算）。
- 預算是辨識＋翻譯合計；用完時會暫停雲端辨識與翻譯。
- 換模型時**一定要同步修改該服務子區段的單價**，否則預算計算會不準。
- 雲端辨識預設不做中間結果（每 0.7 秒重送整段音訊，費用約變 4 倍）；需要時設 `asr.partial_results = "on"`。

## 6. 延遲與調校

### 怎麼調整：先看症狀，一次只改一項

1. 開一場會議或影片實際跑 10 分鐘，記下哪裡不滿意。
2. 從下面的表找到對應的症狀，**一次只改一項**，再跑一次比較。
3. 想看每一句實際花多久：用 `scripts\run.ps1 --log-level DEBUG` 執行，結束後看 `logs\livesub.log`。
4. 改壞了也沒關係：刪掉 `config.toml`，下次啟動會從 `config.example.toml` 重新複製一份預設值。

| 想要 | 在設定視窗 | 設定檔裡的名稱 | 代價 |
|---|---|---|---|
| 字幕更快出現 | 停頓多久算一句 → 400～500 | `segmenter.end_silence_ms` | 句子容易被切碎 |
| 句子更完整、翻譯更通順 | 停頓多久算一句 → 800～1000 | `segmenter.end_silence_ms` | 要多等一下 |
| 講到一半就看到中文 | 勾「講到一半就先翻譯」 | `translate.partial_translation` | 翻譯費用約 4 倍 |
| 最省錢 | 不勾「講到一半就先翻譯」、翻譯選 Haiku 或 gpt-4o-mini | — | 每句講完才出中文 |
| 專有名詞翻得一致 | 「編輯術語表」加上固定譯法 | `glossary.csv` | 無 |
| 專有名詞**聽**得更準 | （進階） | `asr.glossary_prompt = true` | 偶爾會把術語硬塞進句子 |
| 前後用詞更一致 | 參考前文 → 4～5 句 | `translate.context_turns` | 費用略增 |
| 翻譯品質最好 | 翻譯選 Claude Opus 5.5 | — | 費用約 60 倍 |
| 語言判斷更準（中日混雜） | 語音辨識選 whisper-1 或 Groq | — | 價格不同 |
| 不要誤判成其他語言 | 只勾實際會出現的語言 | `asr.allowed_languages` | 沒勾的語言會被略過 |

**術語表（glossary.csv）**：每行一組「原文,固定譯法」，例如 `section control,區間測速`。
用 Excel 編輯時請「另存為 CSV UTF-8」。翻譯時會優先使用這些譯法。

### 參考數值

| 項目 | GPU 下的典型值 |
|---|---|
| 開口後第一次出現原文 | 本機 GPU 約 1 秒；雲端在句子結束後約 1～2 秒 |
| 中間譯文 | 原文出現後再約 0.5～1.5 秒 |
| 定稿 | 講者停頓 0.6 秒後，再約 1～2 秒 |

| 問題 | 調整 |
|---|---|
| 字幕閃得太頻繁 | `partial_interval_s` 調高到 1.0～1.5，或 `partial_translation = false` |
| 一句話被切成好幾段 | `end_silence_ms` 調高到 800～1000 |
| 背景音樂被當成語音 | `vad_aggressiveness = 3` |
| 出現「辨識速度跟不上」 | 本機改用 `medium`／`small`；雲端改用 Groq 或調低 `asr.cloud.timeout_s` |
| 專有名詞常被聽錯 | 加進 `glossary.csv`，並試 `glossary_prompt = true` |
| 日韓短句被判成英文 | 調高 `sticky_language_prob`（短句沿用上一句語言） |

## 7. 疑難排解

- **程式啟動後馬上閃退，log 出現 `cudnn_ops64_9.dll` 或 `cublas64_12.dll`**：
  重新執行 `setup.ps1 -Gpu`；或在設定檔把 `asr.local.device` 改成 `"cpu"`，或改用 `asr.engine = "cloud"`。
- **完全沒有字幕**：確認聲音是從「預設播放裝置」出來的（Teams 可以另外指定喇叭）；
  用 `-ListDevices` 找到正確的 loopback 裝置，填進 `[audio] device_index`。
- **辨識或翻譯一直失敗**：確認對應的金鑰（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`…）是用 `setx` 設定，並**重新開啟**終端機或重新登入。
- **舊的 config.toml**：舊格式（`[translate] model = ...`）會自動對應到新子區段，並在 log 提示；建議參考 `config.example.toml` 更新。
- **字幕視窗點不到**：可能開啟了點擊穿透，請從系統匣選單關閉。
- 所有錯誤都會寫進 `logs\livesub.log`。

## 8. 專案結構

每個功能一個子套件；雲端服務各自一個檔案，由 `registry.py` 依設定建立。

```
live-subtitle/
├─ .venv/                    虛擬環境（setup.ps1 建立，不進 Git）
├─ src/livesub/
│  ├─ app.py                 進入點：命令列參數、組裝元件
│  ├─ pipeline.py            執行緒與佇列（擷取 → 切句 → 辨識 → 翻譯 → 輸出）
│  ├─ config/
│  │  ├─ schema.py           設定結構：每個模組／每個服務一個 dataclass
│  │  ├─ loader.py           TOML 讀取、未知鍵報錯、舊格式相容、路徑與金鑰工具
│  │  ├─ writer.py           設定視窗存檔：只改動變更的值，保留註解
│  │  └─ user_env.py         讀寫 Windows 使用者環境變數（API 金鑰）
│  ├─ core/
│  │  ├─ models.py           管線中傳遞的資料結構
│  │  ├─ budget.py           每月雲端費用帳本（辨識＋翻譯）
│  │  ├─ glossary.py         術語表（翻譯提示與辨識提示共用）
│  │  └─ http.py             雲端 REST 共用：逾時、重試、錯誤訊息
│  ├─ audio/
│  │  ├─ capture.py          WASAPI loopback 擷取、重新取樣、裝置重連
│  │  └─ segmenter.py        VAD 切句與中間快照
│  ├─ asr/
│  │  ├─ base.py             Transcriber 介面、語言判斷與黏著
│  │  ├─ cloud_openai.py     雲端：OpenAI 相容 /audio/transcriptions（OpenAI、Groq…）
│  │  ├─ local_whisper.py    本機：faster-whisper、GPU→CPU 自動降級
│  │  ├─ filters.py          Whisper 幻覺過濾
│  │  ├─ gpu.py              Windows 載入 pip 版 CUDA DLL
│  │  └─ registry.py         依 asr.engine 建立辨識器
│  ├─ translation/
│  │  ├─ base.py             Translator 介面、提示詞、上下文、OpenCC 後處理
│  │  ├─ openai_compat.py    OpenAI 相容 /chat/completions
│  │  ├─ claude.py           Anthropic Claude（官方 SDK）
│  │  ├─ deepl.py            DeepL
│  │  ├─ passthrough.py      不翻譯
│  │  └─ registry.py         依 translate.engine 建立翻譯器
│  ├─ output/transcript.py   SRT／TXT／JSONL 輸出
│  └─ ui/
│     ├─ overlay.py          PyQt6 字幕浮層與系統匣
│     ├─ settings.py         設定視窗（四個分頁）
│     └─ presets.py          設定視窗的服務選項（要加新選項改這裡）
├─ tests/                    pytest（不需 GPU、音效卡、網路）
├─ scripts/                  setup.ps1、run.ps1、build_exe.ps1
├─ packaging/                PyInstaller spec
├─ config.example.toml
└─ glossary.example.csv
```

### 新增一個雲端服務

以新增翻譯服務 `myprovider` 為例（語音辨識同理，換成 `asr/`）：

1. `config/schema.py`：新增 `MyProviderConfig`，掛到 `TranslateConfig.myprovider`，並把 `"myprovider"` 加進 `engine` 的允許值。
2. `translation/myprovider.py`：實作 `translate()`、`warmup()`、`name`、`is_cloud`。
3. `translation/registry.py`：在 `_BUILDERS` 加一行。
4. `config.example.toml`：加上 `[translate.myprovider]` 區段（測試會檢查範例檔與預設值一致）。

開發：

```powershell
scripts\setup.ps1 -Dev            # 要測本機辨識再加 -LocalAsr 或 -Gpu
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check src tests
```

## 9. 已知限制

- 多人同時說話時只會辨識出較大聲的一方，也不會標示是誰在講。
- 中間結果是重新辨識整段音訊的結果，前面的字偶爾會被修正，這是「幾乎同步」的必然代價。
- `gpt-4o-*-transcribe` 不回傳語言，程式以文字系統（假名／諺文／漢字／拉丁字母）判斷；只有漢字的日文短句可能被判成中文。需要更準的語言判斷可改用 `whisper-1` 或 Groq Whisper。
- DeepL 引擎不使用 `glossary.csv`（DeepL 的術語表要在 DeepL 端另外建立）。
- 雲端模式會把**會議音訊**送到語音辨識服務、把**文字**送到翻譯服務，請確認符合公司的資安規範；
  需要完全不外傳時，用 `asr.engine = "local"` 並把 `translate.openai.base_url` 改成本機 Ollama。
