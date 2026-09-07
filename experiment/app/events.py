"""サーバー内イベントバス(SSE へ配信する)。

publish() は同期・非ブロッキング。購読者ごとに Queue(200) を持ち、
溢れたら古いものから捨てる(遅いブラウザがサーバーを詰まらせない)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, maxsize: int = 200) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self._maxsize = maxsize
        self.history_limit = 200
        self.recent: list[dict[str, Any]] = []  # 直近のイベント(UI の再接続時にログ欄を復元する用)

    def publish(self, type_: str, **data: Any) -> dict[str, Any]:
        ev = {"type": type_, "time": time.time(), **data}
        if type_ in ("toast", "log", "connection", "recovery", "error"):
            self.recent.append(ev)
            del self.recent[: -self.history_limit]
        for q in list(self._subs):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass
        return ev

    def toast(self, level: str, message: str) -> None:
        log.log(logging.ERROR if level == "error" else logging.INFO, "toast[%s] %s", level, message)
        self.publish("toast", level=level, message=message)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


def sse_format(ev: dict[str, Any]) -> str:
    return f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"


async def sse_stream(bus: EventBus, snapshot: dict[str, Any], heartbeat: Any, interval: float = 3.0) -> AsyncIterator[str]:
    """SSE 用のジェネレータ。最初に snapshot、以後イベントと定期ハートビート。

    heartbeat: 呼ぶと dict を返す関数(server_time などを載せる)。
    """
    q = bus.subscribe()
    try:
        yield sse_format({"type": "snapshot", "time": time.time(), **snapshot})
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=interval)
                yield sse_format(ev)
            except asyncio.TimeoutError:
                yield sse_format({"type": "heartbeat", "time": time.time(), **heartbeat()})
    finally:
        bus.unsubscribe(q)
