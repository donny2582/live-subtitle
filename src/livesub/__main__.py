"""允許以 `python -m livesub` 執行。

scripts\\run.ps1 就是用這個方式啟動；實際的程式流程在 app.py 的 main()。
打包成 .exe 時，PyInstaller 也是以這個檔案作為入口。
"""

import logging
import sys

from livesub.app import _fatal_dialog, main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.getLogger("livesub").exception("程式發生未預期的錯誤")
        # 從桌面捷徑（pythonw）或 .exe 啟動時沒有主控台，錯誤訊息要用對話框顯示，否則使用者什麼都看不到
        if sys.stderr is None or getattr(sys, "frozen", False):
            _fatal_dialog(f"程式發生錯誤：\n{exc}\n\n詳細內容請看 logs\\livesub.log")
        raise
