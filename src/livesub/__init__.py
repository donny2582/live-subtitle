"""livesub：Windows 即時語音翻譯字幕（英／日／韓 → 繁體中文）。

各子套件的分工（詳見 docs/程式架構說明.md）：
    app.py        進入點，組裝所有元件
    pipeline.py   背景執行緒與佇列：擷取 → 切句 → 辨識 → 翻譯 → 輸出
    config/       設定檔讀寫、API 金鑰
    audio/        收音與切句
    asr/          語音辨識（聲音 → 文字）
    translation/  翻譯（文字 → 繁體中文）
    core/         共用工具：資料結構、費用帳本、術語表、HTTP
    output/       逐字稿檔案
    ui/           字幕浮層、系統匣、設定視窗
"""

__version__ = "0.1.0"
