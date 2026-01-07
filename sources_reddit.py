"""
Простой модуль для получения мемов с Reddit (r/memes) и поиска похожих
в базе мемов Telegram-канала.

Используем публичный JSON-эндпоинт Reddit без авторизации.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import easyocr
import numpy as np
import requests
from PIL import Image

from config import load_app_config
from embeddings import search_similar_by_image
from taste_model import get_taste_model, SIMILARITY_THRESHOLD


@dataclass
class RedditMemeCandidate:
    reddit_id: str
    title: str
    image_url: str
    best_score: float
    best_match: dict


CANDIDATES_FILE = Path("data/reddit_candidates.json")


def _fetch_reddit_json(subreddit: str, limit: int) -> dict:
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"limit": limit, "t": "day"}
    headers = {"User-Agent": "memes-ai-bot/0.1"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _extract_image_posts(data: dict) -> List[dict]:
    posts: List[dict] = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        
        # Получаем URL картинки
        url = post.get("url_overridden_by_dest") or post.get("url")
        if not url:
            continue
        
        # Проверяем, что это картинка (по расширению или домену)
        url_lower = url.lower()
        is_image = (
            any(url_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"))
            or "i.redd.it" in url_lower
            or "i.imgur.com" in url_lower
            or post.get("post_hint") == "image"
        )
        
        # Пропускаем видео и другие типы
        if post.get("post_hint") == "hosted:video":
            continue
        
        if not is_image:
            continue
        
        posts.append(
            {
                "id": post.get("id"),
                "title": post.get("title", ""),
                "url": url,
            }
        )
    return posts


def _download_image(url: str, tmp_dir: Path) -> Path | None:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        suffix = ".jpg"
        if ".png" in url:
            suffix = ".png"
        elif ".webp" in url:
            suffix = ".webp"
        tmp_path = tmp_dir / f"img_{abs(hash(url))}{suffix}"
        tmp_path.write_bytes(resp.content)
        return tmp_path
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка загрузки {url}: {exc}")
        return None


def _extract_text_from_image(image_path: Path) -> str:
    """
    Извлекает текст из изображения с помощью OCR.
    """
    try:
        reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        image = Image.open(image_path).convert("RGB")
        ocr_result = reader.readtext(np.array(image), detail=0)
        return " ".join(ocr_result).strip()
    except Exception:  # noqa: BLE001
        return ""


def fetch_and_match_reddit_memes(
    limit: int = 50,
    use_taste_model: bool = True,
    taste_threshold: float = SIMILARITY_THRESHOLD,
) -> List[RedditMemeCandidate]:
    """
    Загружает мемы из r/memes, ищет для каждого самые похожие мемы
    в базе Telegram-канала и сохраняет кандидатов в JSON.
    
    Args:
        limit: Количество постов для обработки
        use_taste_model: Использовать модель вкуса для фильтрации
        taste_threshold: Порог похожести для модели вкуса
    """
    cfg = load_app_config()
    subreddit = cfg.reddit.subreddit

    print(f"Запрашиваю Reddit r/{subreddit} (лимит: {limit})...")
    try:
        raw = _fetch_reddit_json(subreddit, limit=limit)
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка запроса к Reddit: {exc}")
        return []
    
    posts = _extract_image_posts(raw)
    print(f"Найдено {len(posts)} постов с картинками")
    if not posts:
        print("Не найдено подходящих постов Reddit.")
        return []
    
    # Инициализация модели вкуса, если используется
    taste_model = None
    if use_taste_model:
        try:
            taste_model = get_taste_model()
            if taste_model.taste_vector is None:
                print("⚠️  Вектор вкуса не загружен, фильтрация по вкусу отключена")
                use_taste_model = False
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Ошибка загрузки модели вкуса: {exc}, продолжаю без неё")
            use_taste_model = False

    candidates: List[RedditMemeCandidate] = []

    print(f"Обрабатываю {len(posts)} постов...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx, post in enumerate(posts, 1):
            print(f"[{idx}/{len(posts)}] Обрабатываю: {post['title'][:50]}...")
            img_path = _download_image(post["url"], tmp_dir)
            if not img_path:
                print(f"  ⚠️  Не удалось загрузить картинку")
                continue

            # Загрузка изображения для анализа
            image = Image.open(img_path).convert("RGB")
            
            # Фильтрация по модели вкуса (если включена)
            if use_taste_model and taste_model:
                try:
                    # Извлечение текста из изображения (OCR)
                    ocr_text = _extract_text_from_image(img_path)
                    if not ocr_text:
                        ocr_text = post.get("title", "")
                    
                    # Оценка мема моделью вкуса
                    embedding, similarity, is_similar = taste_model.evaluate_meme(
                        image, ocr_text, taste_threshold
                    )
                    
                    if not is_similar:
                        print(
                            f"  🚽 ПРОПУЩЕН по вкусу (similarity: {similarity:.4f} < {taste_threshold})"
                        )
                        continue
                    else:
                        print(
                            f"  ✅ ОДОБРЕНО по вкусу (similarity: {similarity:.4f})"
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"  ⚠️  Ошибка оценки вкуса: {exc}, пропускаю")
                    continue

            # Поиск похожих мемов в базе канала (опционально, если индекс построен)
            best_score = similarity if use_taste_model else 0.0
            best_meta = {}
            
            try:
                matches = search_similar_by_image(img_path, top_k=3)
                if matches:
                    best_score, best_meta = matches[0]
                    print(f"  ✓ Найдено похожих мемов в базе, лучший score: {best_score:.3f}")
                else:
                    print(f"  ℹ️  Индекс не построен, используем только модель вкуса")
            except FileNotFoundError:
                # Индекс не построен - это нормально, используем только модель вкуса
                print(f"  ℹ️  Индекс не построен, используем только модель вкуса")
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️  Ошибка поиска в базе: {exc}, продолжаем")
            
            candidate = RedditMemeCandidate(
                reddit_id=post["id"],
                title=post["title"],
                image_url=post["url"],
                best_score=best_score,
                best_match=best_meta,
            )
            candidates.append(candidate)

    CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "reddit_id": c.reddit_id,
            "title": c.title,
            "image_url": c.image_url,
            "best_score": c.best_score,
            "best_match": c.best_match,
        }
        for c in candidates
    ]
    CANDIDATES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    print(
        f"Сохранено {len(candidates)} кандидатов в {CANDIDATES_FILE}. "
        "Можно использовать их для дальнейшей отправки в канал."
    )

    return candidates


if __name__ == "__main__":
    fetch_and_match_reddit_memes()


