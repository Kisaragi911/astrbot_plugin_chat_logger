import hashlib
import os
import re
import time
from datetime import datetime

import aiofiles

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter


class Main(star.Star):
    # 去重窗口：同一用户同一平台同一消息内容在 N 秒内只记录一次
    _DEDUP_TTL: float = 30.0
    # 缓存超过此数量时触发清理，防止内存无限增长
    _DEDUP_CLEANUP_THRESHOLD: int = 500

    def __init__(self, context: star.Context) -> None:
        self.context = context
        self.logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        # 去重缓存: key -> 最后写入时间戳
        self._dedup_cache: dict[str, float] = {}

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名中不允许的字符（\/:*?"<>| → 替换为下划线）"""
        return re.sub(r'[\\/:*?"<>|]', "_", name)

    def _get_log_path(self, user_id: str, platform_name: str) -> str:
        safe_uid = self._sanitize_filename(user_id)
        safe_platform = self._sanitize_filename(platform_name)
        return os.path.join(self.logs_dir, f"{safe_uid}_{safe_platform}.txt")

    async def _append_log(
        self, filepath: str, speaker: str, message: str
    ) -> None:
        """异步追加一行日志，不阻塞事件循环"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{speaker}]: {message}\n"
        try:
            async with aiofiles.open(filepath, "a", encoding="utf-8") as f:
                await f.write(line)
        except Exception:
            pass  # 静默失败，绝不因日志写入错误影响 AstrBot 运行

    async def _log_message(
        self, event: AstrMessageEvent, speaker: str, text: str
    ) -> None:
        """统一入口：根据 event 提取元数据并写入日志"""
        if not text or not text.strip():
            return
        user_id = event.get_sender_id()
        platform = event.get_platform_name()
        log_path = self._get_log_path(user_id, platform)
        await self._append_log(log_path, speaker, text)

    # ── 去重 ──────────────────────────────────────────────────

    @staticmethod
    def _dedup_key(user_id: str, platform: str, text: str) -> str:
        """生成去重键：对 (user_id, platform, text) 做稳定哈希"""
        raw = f"{user_id}|{platform}|{text.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_duplicate(self, key: str) -> bool:
        """检查 key 是否在去重窗口内已存在；若不存在则登记"""
        now = time.time()
        last_ts = self._dedup_cache.get(key)
        if last_ts is not None and (now - last_ts) < self._DEDUP_TTL:
            return True
        self._dedup_cache[key] = now
        return False

    def _maybe_cleanup_dedup_cache(self) -> None:
        """定期清理过期条目，控制内存"""
        if len(self._dedup_cache) <= self._DEDUP_CLEANUP_THRESHOLD:
            return
        now = time.time()
        expired = [
            k for k, ts in self._dedup_cache.items()
            if now - ts >= self._DEDUP_TTL
        ]
        for k in expired:
            del self._dedup_cache[k]

    # ── 事件钩子 ──────────────────────────────────────────────

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def on_private_message(self, event: AstrMessageEvent) -> None:
        """捕获用户发来的私聊消息"""
        # 双重保险：再次确认是私聊才处理
        if not event.is_private_chat():
            return

        user_msg = event.get_message_str()
        if not user_msg or not user_msg.strip():
            return

        # 去重：同一用户+平台+消息内容在 30s 内只记录一次
        dedup_key = self._dedup_key(
            event.get_sender_id(),
            event.get_platform_name(),
            user_msg,
        )
        if self._is_duplicate(dedup_key):
            return
        self._maybe_cleanup_dedup_cache()

        await self._log_message(event, event.get_sender_id(), user_msg)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response) -> None:
        """捕获 LLM 回复，仅记录私聊下的最终文本回复"""
        # 只处理私聊
        if not event.is_private_chat():
            return
        # 跳过工具调用响应和流式分块
        if getattr(response, "role", None) != "assistant":
            return
        if getattr(response, "is_chunk", False):
            return

        text = getattr(response, "completion_text", None)
        if isinstance(text, str):
            text = text.strip()
        else:
            return

        await self._log_message(event, "Bot", text)