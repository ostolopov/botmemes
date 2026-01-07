"""
Модуль для отправки кандидатов в Telegram-канал с отложенной отправкой.
Поддерживает отправку в отложку (scheduled messages) с заданным интервалом.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from telethon import TelegramClient

from config import load_app_config
from sent_tracker import filter_sent_memes, add_sent_meme, is_meme_sent

# Файл кандидатов из Reddit
REDDIT_CANDIDATES_FILE = Path("data/reddit_candidates.json")


def _load_candidates() -> List[dict]:
    """
    Загружает кандидатов из Reddit.
    """
    if not REDDIT_CANDIDATES_FILE.exists():
        print(
            f"Файл кандидатов {REDDIT_CANDIDATES_FILE} не найден. "
            "Сначала выполните команду fetch_reddit."
        )
        return []
    
    data = json.loads(REDDIT_CANDIDATES_FILE.read_text("utf-8"))
    print(f"Загружено {len(data)} кандидатов из Reddit")
    return data


async def _post_candidates_async(
    max_count: int = 10,
    interval_hours: float = 1.0,
    start_delay_minutes: int = 0,
) -> None:
    """
    Асинхронная отправка кандидатов в канал через аккаунт пользователя (Telethon)
    с отложенной отправкой (scheduled messages).
    
    Args:
        max_count: Максимальное количество мемов для отправки
        interval_hours: Интервал между постами в часах (по умолчанию 1 час)
        start_delay_minutes: Задержка перед отправкой первого поста в минутах
    """
    cfg = load_app_config()
    channel = cfg.telegram.channel_username

    if not channel:
        print("TELEGRAM_CHANNEL_USERNAME не задан.")
        return

    candidates = _load_candidates()
    if not candidates:
        return

    # Фильтруем уже отправленные мемы
    candidates = filter_sent_memes(candidates)
    if not candidates:
        print("Все мемы уже были отправлены. Нужно найти новые мемы.")
        return

    to_send = candidates[:max_count]
    print(f"Будет отправлено {len(to_send)} мемов в отложку (после фильтрации дубликатов)")

    client = TelegramClient(
        "session_scheduler",
        cfg.telegram.api_id,
        cfg.telegram.api_hash,
    )

    async with client:
        # Время первого поста
        current_time = datetime.now()
        if start_delay_minutes > 0:
            schedule_time = current_time + timedelta(minutes=start_delay_minutes)
        else:
            # Если задержка не указана, отправляем первый пост через 5 минут
            schedule_time = current_time + timedelta(minutes=5)
        
        first_post_time = schedule_time
        interval = timedelta(hours=interval_hours)
        
        for idx, cand in enumerate(to_send, 1):
            caption = cand.get("title", "")
            best = cand.get("best_match") or {}
            orig_caption = best.get("caption") or ""

            full_caption = caption
            if orig_caption:
                full_caption += (
                    f"\n\n(Похоже на пост из канала: {orig_caption[:200]})"
                )

            image_url = cand["image_url"]
            meme_id = cand.get("reddit_id") or cand.get("image_url", "unknown")
            
            # Проверяем, не был ли мем уже отправлен (дополнительная проверка)
            if is_meme_sent(meme_id):
                print(f"[{idx}/{len(to_send)}] Мем {meme_id} уже был отправлен, пропускаю")
                continue
            
            try:
                # Отправляем в отложку
                await client.send_message(
                    entity=channel,
                    message=full_caption[:1024],
                    file=image_url,
                    schedule=schedule_time,
                )
                print(
                    f"[{idx}/{len(to_send)}] Мем {meme_id} запланирован на "
                    f"{schedule_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                # Сохраняем ID отправленного мема
                add_sent_meme(meme_id)
                
                # Увеличиваем время для следующего поста
                schedule_time += interval
                
            except Exception as exc:  # noqa: BLE001
                print(f"Ошибка отправки {meme_id}: {exc}")
        
        last_post_time = schedule_time - interval
        print(f"\n✓ Все мемы запланированы в отложку канала {channel}")
        print(f"  Первый пост: {first_post_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Последний пост: {last_post_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Интервал между постами: {interval_hours} часов")


def post_candidates_to_channel(
    max_count: int = 10,
    interval_hours: float = 1.0,
    start_delay_minutes: int = 0,
) -> None:
    """
    Синхронная обёртка для удобного запуска из main.py.
    
    Args:
        max_count: Максимальное количество мемов для отправки
        interval_hours: Интервал между постами в часах (по умолчанию 1 час)
        start_delay_minutes: Задержка перед отправкой первого поста в минутах
    """
    try:
        # Проверяем, запущен ли event loop
        asyncio.get_running_loop()
        # Если loop запущен, нельзя использовать asyncio.run()
        # В этом случае нужно вызывать _post_candidates_async напрямую через await
        raise RuntimeError(
            "post_candidates_to_channel() вызвана из async контекста. "
            "Используйте await _post_candidates_async() напрямую."
        )
    except RuntimeError as e:
        if "get_running_loop" in str(e):
            # Event loop не запущен, используем asyncio.run()
            asyncio.run(
                _post_candidates_async(
                    max_count=max_count,
                    interval_hours=interval_hours,
                    start_delay_minutes=start_delay_minutes,
                )
            )
        else:
            raise


if __name__ == "__main__":
    post_candidates_to_channel()


