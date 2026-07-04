"""
API Key 驗證中介層。

設定 .env 的 API_KEY 後，受保護路由需在請求帶上 X-API-Key header。
留空（預設）則不驗證，僅建議本機開發環境使用；對外部署務必設定 API_KEY。
"""
from __future__ import annotations

import logging

from fastapi import Header, HTTPException, status

from core.config import settings

logger = logging.getLogger(__name__)

_warned = False


async def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    global _warned
    if not settings.api_key:
        if not _warned:
            logger.warning(
                "[Auth] API_KEY 未設定，管理端點目前不受保護。"
                "對外部署前請務必在 .env 設定 API_KEY。"
            )
            _warned = True
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效或缺少 X-API-Key",
        )
