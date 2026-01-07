"""
Модуль для мониторинга других Telegram-каналов и копирования их постов.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import List, Set

import easyocr
import numpy as np
from PIL import Image
from telethon import TelegramClient
from telethon.tl.types import Message, MessageMediaPhoto, MessageMediaDocument

from config import load_app_config
from sent_tracker import add_sent_meme
from taste_model import get_taste_model, SIMILARITY_THRESHOLD

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


def _extract_text_from_image(image_data: bytes) -> str:
    """
    Извлекает текст из изображения с помощью OCR.
    
    Args:
        image_data: Байты изображения
    
    Returns:
        Извлечённый текст
    """
    try:
        reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        ocr_result = reader.readtext(np.array(image), detail=0)
        return " ".join(ocr_result).strip()
    except Exception:  # noqa: BLE001
        return ""


async def monitor_telegram_channels(
    source_channels: List[str],
    target_channel: str,
    schedule_delay_hours: int = 1,
    limit_per_channel: int = 10,
    use_taste_model: bool = True,
    taste_threshold: float = SIMILARITY_THRESHOLD,
) -> int:
    """
    Мониторит указанные Telegram-каналы и копирует новые посты (только медиа) в отложку целевого канала.
    Фильтрует посты по модели вкуса.
    
    Args:
        source_channels: Список @username каналов для мониторинга
        target_channel: @username целевого канала
        schedule_delay_hours: Задержка перед публикацией (в часах)
        limit_per_channel: Максимум постов для проверки с каждого канала
        use_taste_model: Использовать модель вкуса для фильтрации
        taste_threshold: Порог похожести для модели вкуса
    
    Returns:
        Количество скопированных постов
    """
    cfg = load_app_config()
    processed = load_processed_messages()
    
    # Инициализация модели вкуса, если используется
    taste_model = None
    if use_taste_model:
        try:
            taste_model = get_taste_model()
            if taste_model.taste_vector is None:
                logger.warning("Вектор вкуса не загружен, фильтрация по вкусу отключена")
                use_taste_model = False
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Ошибка загрузки модели вкуса: {exc}, продолжаю без неё")
            use_taste_model = False
    
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
                
                # Получаем последние сообщения из канала (только с медиа)
                messages = []
                async for msg in client.iter_messages(
                    channel_username, limit=limit_per_channel
                ):
                    # Используем уникальный ID для каждого канала
                    unique_msg_id = f"{channel_username}_{msg.id}"
                    if unique_msg_id in processed:
                        continue
                    # Фильтруем только посты с медиа (фото или видео)
                    if not msg.media:
                        continue
                    
                    # Проверяем тип медиа: фото или видео
                    is_photo = isinstance(msg.media, MessageMediaPhoto)
                    is_video = False
                    if isinstance(msg.media, MessageMediaDocument):
                        doc = msg.media.document
                        if doc and hasattr(doc, 'mime_type'):
                            mime_type = doc.mime_type or ""
                            is_video = mime_type.startswith("video/")
                    
                    if not (is_photo or is_video):
                        continue
                    
                    messages.append(msg)
                
                logger.info(f"Найдено {len(messages)} постов с медиа в {channel_username}")
                
                for msg in messages:
                    # Используем уникальный ID для каждого канала
                    unique_msg_id = f"{channel_username}_{msg.id}"
                    
                    # Проверяем, не был ли уже обработан
                    if unique_msg_id in processed:
                        continue
                    
                    try:
                        # Фильтрация по модели вкуса (только для фото, видео пропускаем без проверки)
                        is_photo_media = isinstance(msg.media, MessageMediaPhoto)
                        if use_taste_model and taste_model and is_photo_media:
                            # Скачиваем изображение во временный файл
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                tmp_path = Path(tmp_file.name)
                            
                            try:
                                downloaded_path = await msg.download_media(file=str(tmp_path))
                                if not downloaded_path or not Path(downloaded_path).exists():
                                    logger.warning(f"Не удалось скачать медиа для поста {msg.id}")
                                    processed.add(unique_msg_id)
                                    add_processed_message(unique_msg_id)
                                    continue
                                
                                # Загружаем изображение
                                image = Image.open(downloaded_path).convert("RGB")
                                
                                # Извлекаем текст из изображения (OCR)
                                with open(downloaded_path, 'rb') as f:
                                    img_bytes = f.read()
                                ocr_text = _extract_text_from_image(img_bytes)
                                
                                # Оценка мема моделью вкуса
                                embedding, similarity, is_similar = taste_model.evaluate_meme(
                                    image, ocr_text, taste_threshold
                                )
                                
                                if not is_similar:
                                    logger.info(
                                        f"  🚽 ПРОПУЩЕН по вкусу (similarity: {similarity:.4f} < {taste_threshold})"
                                    )
                                    # Помечаем как обработанное, чтобы не проверять снова
                                    processed.add(unique_msg_id)
                                    add_processed_message(unique_msg_id)
                                    # Удаляем временный файл
                                    Path(downloaded_path).unlink(missing_ok=True)
                                    continue
                                else:
                                    logger.info(
                                        f"  ✅ ОДОБРЕНО по вкусу (similarity: {similarity:.4f})"
                                    )
                                
                                # Удаляем временный файл
                                Path(downloaded_path).unlink(missing_ok=True)
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(f"Ошибка при фильтрации по вкусу: {exc}, пропускаю")
                                Path(downloaded_path).unlink(missing_ok=True) if 'downloaded_path' in locals() else None
                                processed.add(unique_msg_id)
                                add_processed_message(unique_msg_id)
                                continue
                        
                        # Отправляем только медиа БЕЗ текста
                        await client.send_message(
                            entity=target_channel,
                            message="",  # Без текста, только медиа
                            file=msg.media,
                            schedule=schedule_time,
                        )
                        
                        # Сохраняем как обработанное (используем уникальный ID)
                        processed.add(unique_msg_id)
                        add_processed_message(unique_msg_id)
                        # Также добавляем в sent_tracker
                        add_sent_meme(unique_msg_id)
                        
                        logger.info(
                            f"Скопирован мем {msg.id} из {channel_username} "
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

