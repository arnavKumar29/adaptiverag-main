"""
JWT authentication + API key validation + Redis rate limiter.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import get_settings
from api.db.postgres import ApiKey, get_db

logger = logging.getLogger(__name__)
settings = get_settings()

security = HTTPBearer(auto_error=False)

# ── Redis client (module singleton) ───────────────────────────────────────────
_redis: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=settings.jwt_expire_hours)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Rate limiter ──────────────────────────────────────────────────────────────
async def check_rate_limit(identifier: str) -> None:
    """Sliding window rate limiter: max 100 requests/minute per identifier."""
    redis = get_redis()
    key = f"rate_limit:{identifier}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, 60)
    if count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {settings.rate_limit_per_minute} req/min",
        )


# ── API key validation ────────────────────────────────────────────────────────
async def validate_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[str]:
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return None
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )
    return api_key


# ── Combined auth dependency ──────────────────────────────────────────────────
async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    db: AsyncSession = Depends(get_db),
) -> str:
    """
    Accept either:
      - Bearer JWT token (Authorization: Bearer <token>)
      - API key (X-API-Key: <key>)
    Also enforces rate limiting.
    """
    identifier: Optional[str] = None

    # Try JWT first
    if credentials and credentials.credentials:
        payload = decode_token(credentials.credentials)
        identifier = payload.get("sub", "jwt-user")

    # Try API key
    if not identifier:
        api_key = await validate_api_key(request, db)
        if api_key:
            identifier = hashlib.sha256(api_key.encode()).hexdigest()[:16]

    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: provide Bearer token or X-API-Key header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Rate limit by identifier
    await check_rate_limit(identifier)
    return identifier
