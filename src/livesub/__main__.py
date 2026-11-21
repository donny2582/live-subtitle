"""允許以 `python -m livesub` 執行。

scripts\\run.ps1 就是用這個方式啟動；實際的程式流程在 app.py 的 main()。
打包成 .exe 時，PyInstaller 也是以這個檔案作為入口。
"""

from livesub.app import main

if __name__ == "__main__":
    raise SystemExit(main())
