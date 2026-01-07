"""
Модуль для мониторинга других Telegram-каналов и копирования их постов.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set

from telethon import TelegramClient
from telethon.tl.types import Message

from config import load_app_config
from sent_tracker import add_sent_meme, is_meme_sent

logger = logging.getLogger(__name__)

# Файл для хранения обработанных сообщений
PROCESSED_MESSAGES_FILE = Path("data/processed_messages.json")


def load_processed_messages() -> Set[int]:
    """
    Загружает множество ID уже обработанных сообщений.
    
    Returns:
        Множество ID обработанных сообщений
    """
    if not PROCESSED_MESSAGES_FILE.exists():
        return set()
    
    try:
        data = json.loads(PROCESSED_MESSAGES_FILE.read_text("utf-8"))
        return set(data.get("message_ids", []))
    except Exception:  # noqa: BLE001
        return set()


def save_processed_messages(message_ids: Set[int]) -> None:
    """
    Сохраняет множество ID обработанных сообщений в файл.
    
    Args:
        message_ids: Множество ID обработанных сообщений
    """
    PROCESSED_MESSAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"message_ids": list(message_ids)}
    PROCESSED_MESSAGES_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
    )


def add_processed_message(message_id: int) -> None:
    """
    Добавляет ID сообщения в список обработанных.
    
    Args:
        message_id: ID сообщения
    """
    processed = load_processed_messages()
    processed.add(message_id)
    save_processed_messages(processed)


async def monitor_telegram_channels(
    source_channels: List[str],
    target_channel: str,
    schedule_delay_hours: int = 1,
    limit_per_channel: int = 10,
) -> int:
    """
    Мониторит указанные Telegram-каналы и копирует новые посты в отложку целевого канала.
    
    Args:
        source_channels: Список @username каналов для мониторинга
        target_channel: @username целевого канала
        schedule_delay_hours: Задержка перед публикацией (в часах)
        limit_per_channel: Максимум постов для проверки с каждого канала
    
    Returns:
        Количество скопированных постов
    """
    cfg = load_app_config()
    processed = load_processed_messages()
    
    client = TelegramClient(
        "session_monitor",
        cfg.telegram.api_id,
        cfg.telegram.api_hash,
    )
    
    copied_count = 0
    
    async with client:
        # Время для отложенной публикации
        schedule_time = datetime.now() + timedelta(hours=schedule_delay_hours)
        interval = timedelta(hours=1)  # Интервал между постами
        
        for channel_username in source_channels:
            try:
                logger.info(f"Мониторинг канала: {channel_username}")
                
                # Получаем последние сообщения из канала
                messages = []
                async for msg in client.iter_messages(
                    channel_username, limit=limit_per_channel
                ):
                    # Используем уникальный ID для каждого канала
                    unique_msg_id = f"{channel_username}_{msg.id}"
                    if unique_msg_id in processed:
                        continue
                    if not msg.media and not msg.message:
                        continue
                    messages.append(msg)
                
                logger.info(f"Найдено {len(messages)} новых постов в {channel_username}")
                
                for msg in messages:
                    # Используем уникальный ID для каждого канала
                    unique_msg_id = f"{channel_username}_{msg.id}"
                    
                    # Проверяем, не был ли уже обработан
                    if unique_msg_id in processed:
                        continue
                    
                    try:
                        # Формируем подпись
                        caption = msg.message or ""
                        if len(caption) > 1024:
                            caption = caption[:1021] + "..."
                        
                        # Отправляем в отложку
                        if msg.media:
                            await client.send_message(
                                entity=target_channel,
                                message=caption,
                                file=msg.media,
                                schedule=schedule_time,
                            )
                        else:
                            await client.send_message(
                                entity=target_channel,
                                message=caption,
                                schedule=schedule_time,
                            )
                        
                        # Сохраняем как обработанное (используем уникальный ID)
                        processed.add(unique_msg_id)
                        add_processed_message(unique_msg_id)
                        # Также добавляем в sent_tracker
                        add_sent_meme(unique_msg_id)
                        
                        logger.info(
                            f"Скопирован пост {msg.id} из {channel_username} "
                            f"на {schedule_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        
                        copied_count += 1
                        schedule_time += interval
                        
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            f"Ошибка копирования поста {msg.id} из {channel_username}: {exc}"
                        )
                        continue
                
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Ошибка мониторинга канала {channel_username}: {exc}")
                continue
        
        # Сохраняем обновлённый список обработанных сообщений
        save_processed_messages(processed)
    
    return copied_count

