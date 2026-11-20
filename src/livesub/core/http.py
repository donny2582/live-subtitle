"""雲端 REST 呼叫共用：逾時、429／5xx 重試、錯誤訊息中文化。"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)


class CloudError(RuntimeError):
    """雲端服務呼叫失敗（網路、金鑰、額度等）。各模組再繼承成自己的錯誤類別。"""


def post_json(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retries: int,
    error_cls: type[CloudError] = CloudError,
    label: str = "雲端服務",
    **kwargs: Any,
) -> dict[str, Any]:
    """POST 並回傳 JSON。429／5xx／連線錯誤會重試；401／403 立即失敗。

    kwargs 直接傳給 session.post（json=…、data=…、files=…）。
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = session.post(url, timeout=timeout, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
            log.debug("%s 請求失敗（第 %d 次）：%s", label, attempt + 1, exc)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last = error_cls(f"HTTP {resp.status_code}")
            time.sleep(0.5 * (attempt + 1))
            continue
        if resp.status_code in (401, 403):
            raise error_cls(f"{label} API 金鑰無效或沒有權限（HTTP {resp.status_code}）")
        if resp.status_code >= 400:
            raise error_cls(f"{label} HTTP {resp.status_code}：{resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise error_cls(f"{label} 回傳不是 JSON") from exc
    raise error_cls(f"{label} 無回應：{last}")


def bearer_session(api_key: str, session: requests.Session | None = None) -> requests.Session:
    s = session or requests.Session()
    if api_key:
        s.headers.update({"Authorization": f"Bearer {api_key}"})
    return s
