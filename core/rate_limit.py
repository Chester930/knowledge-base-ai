"""
簡易記憶體內請求限流（per-source sliding window）。

不依賴外部服務（不需要 Redis），足以擋掉單一來源的暴衝式濫用；
限制：僅適合單一 worker/單機部署，多 worker 或多副本部署時各自獨立計數，
非全域準確限流（若未來擴充多 worker，需改用 Redis 等共享儲存）。
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque

from fastapi import HTTPException, Request, status

from core.config import settings

_WINDOW_SECONDS = 60
_MAX_TRACKED_KEYS = 5000  # 上限追蹤來源數，避免偽造大量來源造成記憶體無限增長

_requests: OrderedDict[str, deque] = OrderedDict()


def _client_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return f"ip:{request.client.host}" if request.client else "ip:unknown"


async def rate_limit(request: Request) -> None:
    limit = settings.chat_rate_limit_per_minute
    if limit <= 0:
        return

    key = _client_key(request)
    now = time.time()

    if key in _requests:
        _requests.move_to_end(key)
        window = _requests[key]
    else:
        if len(_requests) >= _MAX_TRACKED_KEYS:
            _requests.popitem(last=False)
        window = deque()
        _requests[key] = window

    while window and now - window[0] > _WINDOW_SECONDS:
        window.popleft()

    if len(window) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"請求過於頻繁，請稍後再試（限每分鐘 {limit} 次）",
        )
    window.append(now)
