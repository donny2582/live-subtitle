"""允許以 `python -m livesub` 執行。"""

from livesub.app import main

if __name__ == "__main__":
    raise SystemExit(main())
