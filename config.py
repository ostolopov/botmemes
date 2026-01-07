import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


load_dotenv()


@dataclass
class TelegramConfig:
    api_id: int
    api_hash: str
    channel_username: str
    source_channels: List[str]  # Список каналов для мониторинга


@dataclass
class RedditConfig:
    client_id: str | None = None
    client_secret: str | None = None
    user_agent: str | None = None
    subreddit: str = "Pikabu"  # Русскоязычный сабреддит с мемами


@dataclass
class EmbeddingsConfig:
    image_model_name: str = "clip-ViT-B-32"
    text_model_name: str = "all-MiniLM-L6-v2"
    embeddings_dir: str = "data/embeddings"
    media_dir: str = "data/media"
    db_path: str = "data/memes.db"


@dataclass
class AppConfig:
    telegram: TelegramConfig
    reddit: RedditConfig
    embeddings: EmbeddingsConfig


def load_telegram_config() -> TelegramConfig:
    # Парсим список каналов из переменной окружения (через запятую)
    source_channels_str = os.environ.get("SOURCE_TELEGRAM_CHANNELS", "")
    source_channels = [
        ch.strip() for ch in source_channels_str.split(",") if ch.strip()
    ]
    
    # Если список пуст, используем дефолтные каналы
    if not source_channels:
        source_channels = [
            "@why4ch",
            "@beobanka",
            "@saved_2_0",
            "@profunctor_io",
            "@subject282",
            "@poieajpoij",
            "@takoi_sebe_channel",
            "@reptilememe",
            "@aftertitles",
            "@Katzen_und_Politik",
            "@transhumemism",
            "@bogmoegopadika",
        ]
    
    return TelegramConfig(
        api_id=int(os.environ["TELEGRAM_API_ID"]),
        api_hash=os.environ["TELEGRAM_API_HASH"],
        channel_username=os.environ.get("TELEGRAM_CHANNEL_USERNAME", ""),
        source_channels=source_channels,
    )


def load_reddit_config() -> RedditConfig:
    return RedditConfig(
        client_id=os.environ.get("REDDIT_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
        user_agent=os.environ.get("REDDIT_USER_AGENT", "memes-bot/0.1"),
        subreddit=os.environ.get("REDDIT_SUBREDDIT", "memes"),
    )


def load_embeddings_config() -> EmbeddingsConfig:
    return EmbeddingsConfig(
        embeddings_dir=os.environ.get("EMBEDDINGS_DIR", "data/embeddings"),
        media_dir=os.environ.get("MEDIA_DIR", "data/media"),
        db_path=os.environ.get("DB_PATH", "data/memes.db"),
    )


def load_app_config() -> AppConfig:
    return AppConfig(
        telegram=load_telegram_config(),
        reddit=load_reddit_config(),
        embeddings=load_embeddings_config(),
    )


