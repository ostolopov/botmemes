import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from telethon import TelegramClient
from telethon.tl.types import Message

from config import load_app_config


@dataclass
class ExportStats:
    total: int = 0
    with_media: int = 0


async def _handle_message(
    msg: Message, media_dir: Path, stats: ExportStats
) -> None:
    stats.total += 1

    if not msg.media:
        return

    media_dir.mkdir(parents=True, exist_ok=True)

    file_path = await msg.download_media(file=media_dir / f"{msg.id}")
    if not file_path:
        return

    text_path = media_dir / f"{msg.id}.txt"
    caption = (msg.message or "").strip()
    text_path.write_text(caption, encoding="utf-8")

    stats.with_media += 1


async def export_channel_history(
    channel_username: Optional[str] = None,
    limit: Optional[int] = None,
) -> ExportStats:
    """
    Экспортирует историю канала: медиа (картинки/видео) и подписи.
    Файлы сохраняются в директорию media_dir из настроек.
    """
    cfg = load_app_config()
    telegram_cfg = cfg.telegram
    media_dir = Path(cfg.embeddings.media_dir)

    client = TelegramClient(
        "session_export",
        telegram_cfg.api_id,
        telegram_cfg.api_hash,
    )

    stats = ExportStats()

    async with client:
        entity = channel_username or telegram_cfg.channel_username
        async for msg in client.iter_messages(entity, limit=limit):
            await _handle_message(msg, media_dir, stats)

    return stats


def main() -> None:
    """
    CLI-обёртка: экспортирует все сообщения канала.
    Запуск:
        python telegram_export.py
    """
    limit_env = os.environ.get("EXPORT_LIMIT")
    limit = int(limit_env) if limit_env else None

    stats = asyncio.run(export_channel_history(limit=limit))
    print(
        f"Экспорт завершён. Всего сообщений: {stats.total}, "
        f"с медиа: {stats.with_media}"
    )


if __name__ == "__main__":
    main()


