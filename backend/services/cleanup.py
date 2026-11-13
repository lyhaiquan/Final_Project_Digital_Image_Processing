"""Xoá kết quả quét và ảnh tạm quá hạn (QT-09, NFR-B09)."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..db import Scan, utcnow

log = logging.getLogger("calo_khay")


def purge(sessions: sessionmaker, tmp_dir: Path, ttl_hours: float) -> int:
    now = utcnow()
    with sessions() as db:
        old = db.scalars(select(Scan).where(Scan.expires_at < now)).all()
        for s in old:
            if s.image_path:
                Path(s.image_path).unlink(missing_ok=True)
            db.delete(s)
        db.commit()
    # ảnh mồ côi (server tắt giữa chừng...)
    cutoff = time.time() - ttl_hours * 3600
    for f in tmp_dir.glob("*.jpg"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
    if old:
        log.info("Đã xoá %d kết quả quét quá hạn", len(old))
    return len(old)


async def loop(sessions: sessionmaker, tmp_dir: Path, ttl_hours: float, interval_s: int) -> None:
    while True:
        try:
            await asyncio.to_thread(purge, sessions, tmp_dir, ttl_hours)
        except Exception:
            log.exception("Dọn dẹp thất bại")
        await asyncio.sleep(interval_s)
