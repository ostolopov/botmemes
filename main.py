"""
Точка входа для различных команд проекта.

Примеры:
    python main.py export_telegram
    python main.py build_embeddings
    python main.py fetch_reddit
    python main.py schedule_posts
"""

import argparse

from telegram_export import main as export_telegram_main
from embeddings import build_faiss_index
from sources_reddit import fetch_and_match_reddit_memes
from scheduler import post_candidates_to_channel
from auto_mode import AutoMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Memes AI Bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("export_telegram", help="Экспорт истории канала")
    subparsers.add_parser(
        "build_embeddings", help="Построить эмбеддинги и FAISS-индекс"
    )

    fetch_reddit_parser = subparsers.add_parser(
        "fetch_reddit", help="Загрузить мемы с Reddit и найти похожие"
    )
    fetch_reddit_parser.add_argument(
        "--limit", type=int, default=50, help="Сколько постов Reddit обрабатывать"
    )
    fetch_reddit_parser.add_argument(
        "--use-taste",
        action="store_true",
        default=True,
        help="Использовать модель вкуса для фильтрации (по умолчанию включено)",
    )
    fetch_reddit_parser.add_argument(
        "--no-taste",
        action="store_false",
        dest="use_taste",
        help="Отключить фильтрацию по модели вкуса",
    )
    fetch_reddit_parser.add_argument(
        "--taste-threshold",
        type=float,
        default=0.6,
        help="Порог похожести для модели вкуса (по умолчанию 0.6)",
    )

    schedule_parser = subparsers.add_parser(
        "schedule_posts", help="Отправить найденные мемы в канал в отложку"
    )
    schedule_parser.add_argument(
        "--max", type=int, default=10, help="Максимум мемов для отправки"
    )
    schedule_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Интервал между постами в часах (по умолчанию 1.0)",
    )
    schedule_parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Задержка перед отправкой первого поста в минутах (по умолчанию 5)",
    )

    auto_parser = subparsers.add_parser(
        "auto", help="Автономный режим: автоматический поиск и отправка мемов"
    )
    auto_parser.add_argument(
        "--search-interval",
        type=float,
        default=1.0,
        help="Интервал между поисками мемов в часах (по умолчанию 1.0)",
    )
    auto_parser.add_argument(
        "--posts-per-search",
        type=int,
        default=50,
        help="Сколько постов обрабатывать за один поиск (по умолчанию 50)",
    )
    auto_parser.add_argument(
        "--posts-to-schedule",
        type=int,
        default=10,
        help="Сколько мемов отправлять в отложку за раз (по умолчанию 10)",
    )
    auto_parser.add_argument(
        "--schedule-interval",
        type=float,
        default=2.0,
        help="Интервал между постами в отложке в часах (по умолчанию 2.0)",
    )
    auto_parser.add_argument(
        "--schedule-delay",
        type=int,
        default=10,
        help="Задержка перед отправкой первого поста в минутах (по умолчанию 10)",
    )
    auto_parser.add_argument(
        "--taste-threshold",
        type=float,
        default=0.6,
        help="Порог похожести для модели вкуса (по умолчанию 0.6)",
    )

    args = parser.parse_args()

    if args.command == "export_telegram":
        export_telegram_main()
    elif args.command == "build_embeddings":
        build_faiss_index()
    elif args.command == "fetch_reddit":
        fetch_and_match_reddit_memes(
            limit=args.limit,
            use_taste_model=getattr(args, "use_taste", True),
            taste_threshold=getattr(args, "taste_threshold", 0.6),
        )
    elif args.command == "schedule_posts":
        post_candidates_to_channel(
            max_count=args.max,
            interval_hours=args.interval,
            start_delay_minutes=args.delay,
        )
    elif args.command == "auto":
        import asyncio

        auto_mode = AutoMode(
            search_interval_hours=args.search_interval,
            posts_per_search=args.posts_per_search,
            posts_to_schedule=args.posts_to_schedule,
            schedule_interval_hours=args.schedule_interval,
            schedule_delay_minutes=args.schedule_delay,
            taste_threshold=args.taste_threshold,
        )
        asyncio.run(auto_mode.run())
    else:
        parser.error(f"Неизвестная команда: {args.command}")


if __name__ == "__main__":
    main()


