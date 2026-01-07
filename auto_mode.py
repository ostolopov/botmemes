"""
Автономный режим работы бота.
Автоматически ищет мемы с Reddit и отправляет их в отложку канала.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from config import load_app_config
from scheduler import post_candidates_to_channel
from sources_reddit import fetch_and_match_reddit_memes

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("auto_mode.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class AutoMode:
    """
    Автономный режим работы бота.
    Периодически ищет мемы и отправляет их в отложку.
    """

    def __init__(
        self,
        search_interval_hours: float = 1.0,
        posts_per_search: int = 50,
        posts_to_schedule: int = 10,
        schedule_interval_hours: float = 2.0,
        schedule_delay_minutes: int = 10,
        taste_threshold: float = 0.6,
    ) -> None:
        """
        Инициализация автономного режима.

        Args:
            search_interval_hours: Интервал между поисками мемов (в часах)
            posts_per_search: Сколько постов обрабатывать за один поиск
            posts_to_schedule: Сколько мемов отправлять в отложку за раз
            schedule_interval_hours: Интервал между постами в отложке (в часах)
            schedule_delay_minutes: Задержка перед отправкой первого поста (в минутах)
            taste_threshold: Порог похожести для модели вкуса
        """
        self.search_interval_hours = search_interval_hours
        self.posts_per_search = posts_per_search
        self.posts_to_schedule = posts_to_schedule
        self.schedule_interval_hours = schedule_interval_hours
        self.schedule_delay_minutes = schedule_delay_minutes
        self.taste_threshold = taste_threshold
        self.running = False

    async def run_cycle(self) -> None:
        """
        Один цикл работы: поиск мемов и отправка в отложку.
        """
        try:
            logger.info("=" * 60)
            logger.info("Начало цикла поиска и отправки мемов")
            logger.info(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Шаг 1: Поиск мемов с Reddit
            logger.info(f"Поиск мемов с Reddit (лимит: {self.posts_per_search})...")
            candidates = fetch_and_match_reddit_memes(
                limit=self.posts_per_search,
                use_taste_model=True,
                taste_threshold=self.taste_threshold,
            )

            if not candidates:
                logger.warning("Не найдено подходящих мемов. Пропускаю отправку.")
                return

            logger.info(f"Найдено {len(candidates)} подходящих мемов")

            # Шаг 2: Отправка в отложку
            logger.info(
                f"Отправка {self.posts_to_schedule} мемов в отложку "
                f"(интервал: {self.schedule_interval_hours} ч, "
                f"задержка: {self.schedule_delay_minutes} мин)..."
            )
            post_candidates_to_channel(
                max_count=self.posts_to_schedule,
                interval_hours=self.schedule_interval_hours,
                start_delay_minutes=self.schedule_delay_minutes,
            )

            logger.info("✓ Цикл завершён успешно")
            logger.info("=" * 60)

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Ошибка в цикле работы: {exc}", exc_info=True)

    async def run(self) -> None:
        """
        Запуск автономного режима (бесконечный цикл).
        """
        self.running = True
        logger.info("🚀 Автономный режим запущен")
        logger.info(f"Интервал поиска: {self.search_interval_hours} часов")
        logger.info(f"Постов за поиск: {self.posts_per_search}")
        logger.info(f"Мемов в отложку за раз: {self.posts_to_schedule}")
        logger.info(f"Интервал между постами: {self.schedule_interval_hours} часов")
        logger.info("Нажмите Ctrl+C для остановки")

        # Первый запуск сразу
        await self.run_cycle()

        # Затем по расписанию
        while self.running:
            try:
                wait_seconds = int(self.search_interval_hours * 3600)
                logger.info(
                    f"Ожидание {self.search_interval_hours} часов до следующего поиска..."
                )
                await asyncio.sleep(wait_seconds)
                await self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Получен сигнал остановки")
                self.running = False
                break
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Ошибка в основном цикле: {exc}", exc_info=True)
                # Продолжаем работу даже при ошибке
                await asyncio.sleep(300)  # Ждём 5 минут перед повтором

        logger.info("🛑 Автономный режим остановлен")

    def stop(self) -> None:
        """Остановка автономного режима."""
        self.running = False


def main() -> None:
    """
    Точка входа для автономного режима.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Автономный режим работы бота")
    parser.add_argument(
        "--search-interval",
        type=float,
        default=1.0,
        help="Интервал между поисками мемов в часах (по умолчанию 1.0)",
    )
    parser.add_argument(
        "--posts-per-search",
        type=int,
        default=50,
        help="Сколько постов обрабатывать за один поиск (по умолчанию 50)",
    )
    parser.add_argument(
        "--posts-to-schedule",
        type=int,
        default=10,
        help="Сколько мемов отправлять в отложку за раз (по умолчанию 10)",
    )
    parser.add_argument(
        "--schedule-interval",
        type=float,
        default=2.0,
        help="Интервал между постами в отложке в часах (по умолчанию 2.0)",
    )
    parser.add_argument(
        "--schedule-delay",
        type=int,
        default=10,
        help="Задержка перед отправкой первого поста в минутах (по умолчанию 10)",
    )
    parser.add_argument(
        "--taste-threshold",
        type=float,
        default=0.6,
        help="Порог похожести для модели вкуса (по умолчанию 0.6)",
    )

    args = parser.parse_args()

    auto_mode = AutoMode(
        search_interval_hours=args.search_interval,
        posts_per_search=args.posts_per_search,
        posts_to_schedule=args.posts_to_schedule,
        schedule_interval_hours=args.schedule_interval,
        schedule_delay_minutes=args.schedule_delay,
        taste_threshold=args.taste_threshold,
    )

    try:
        asyncio.run(auto_mode.run())
    except KeyboardInterrupt:
        logger.info("Программа остановлена пользователем")


if __name__ == "__main__":
    main()

