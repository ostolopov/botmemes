"""
Модуль для отслеживания уже отправленных мемов.
Предотвращает повторную отправку одного и того же мема.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set

# Файл для хранения истории отправленных мемов
SENT_MEMES_FILE = Path("data/sent_memes.json")


def load_sent_memes() -> Set[str]:
    """
    Загружает множество ID уже отправленных мемов.
    
    Returns:
        Множество ID отправленных мемов
    """
    if not SENT_MEMES_FILE.exists():
        return set()
    
    try:
        data = json.loads(SENT_MEMES_FILE.read_text("utf-8"))
        sent_ids = data.get("sent_ids", [])
        # Обеспечиваем совместимость со старым форматом
        if isinstance(sent_ids, list):
            return set(sent_ids)
        return set()
    except Exception:  # noqa: BLE001
        return set()


def save_sent_memes(sent_ids: Set[str]) -> None:
    """
    Сохраняет множество ID отправленных мемов в файл.
    
    Args:
        sent_ids: Множество ID отправленных мемов
    """
    SENT_MEMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"sent_ids": list(sent_ids)}
    SENT_MEMES_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
    )


def add_sent_meme(meme_id: str) -> None:
    """
    Добавляет ID мема в список отправленных.
    
    Args:
        meme_id: ID мема (например, reddit_id или URL)
    """
    sent_ids = load_sent_memes()
    sent_ids.add(meme_id)
    save_sent_memes(sent_ids)


def is_meme_sent(meme_id: str) -> bool:
    """
    Проверяет, был ли мем уже отправлен.
    
    Args:
        meme_id: ID мема для проверки
    
    Returns:
        True если мем уже был отправлен, False иначе
    """
    sent_ids = load_sent_memes()
    return meme_id in sent_ids


def filter_sent_memes(candidates: list[dict]) -> list[dict]:
    """
    Фильтрует кандидатов, удаляя уже отправленные мемы.
    
    Args:
        candidates: Список кандидатов для фильтрации
    
    Returns:
        Отфильтрованный список кандидатов (без уже отправленных)
    """
    sent_ids = load_sent_memes()
    filtered = []
    
    for candidate in candidates:
        # Используем reddit_id или image_url как уникальный идентификатор
        meme_id = candidate.get("reddit_id") or candidate.get("image_url", "")
        if meme_id and meme_id not in sent_ids:
            filtered.append(candidate)
    
    removed_count = len(candidates) - len(filtered)
    if removed_count > 0:
        print(f"Пропущено {removed_count} уже отправленных мемов")
    
    return filtered

