"""
Модуль для расчёта эмбеддингов мемов и поиска похожих.
Использует CLIP-модель из sentence-transformers и FAISS для поиска.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import load_app_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class MemeRecord:
    id: int
    image_path: Path
    caption: str


class MemeEmbedder:
    def __init__(self, model_name: str) -> None:
        # CLIP-модель может кодировать и текст, и картинки
        self.model = SentenceTransformer(model_name)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        emb = self.model.encode(
            image, convert_to_numpy=True, normalize_embeddings=True
        )
        return emb.astype("float32")

    def encode_caption(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return emb.astype("float32")


def _scan_media_dir(media_dir: Path) -> List[MemeRecord]:
    records: List[MemeRecord] = []
    media_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for path in sorted(media_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        caption_path = path.with_suffix(".txt")
        caption = ""
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()

        records.append(MemeRecord(id=idx, image_path=path, caption=caption))
        idx += 1

    return records


def build_faiss_index() -> None:
    """
    Строит FAISS-индекс по всем мемам из директории media_dir.
    Создаёт файлы:
        - <embeddings_dir>/index.faiss
        - <embeddings_dir>/meta.json
    """
    cfg = load_app_config()
    emb_cfg = cfg.embeddings

    media_dir = Path(emb_cfg.media_dir)
    embeddings_dir = Path(emb_cfg.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    records = _scan_media_dir(media_dir)
    if not records:
        print("Нет медиафайлов для индексации.")
        return

    embedder = MemeEmbedder(emb_cfg.image_model_name)

    all_embeddings: List[np.ndarray] = []
    for rec in records:
        try:
            img = Image.open(rec.image_path).convert("RGB")
            img_emb = embedder.encode_image(img)
            all_embeddings.append(img_emb)
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка при обработке {rec.image_path}: {exc}")

    if not all_embeddings:
        print("Не удалось построить эмбеддинги.")
        return

    mat = np.vstack(all_embeddings).astype("float32")
    dim = mat.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    index_path = embeddings_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    meta = [
        {
            "id": rec.id,
            "image_path": str(rec.image_path),
            "caption": rec.caption,
        }
        for rec in records
    ]
    meta_path = embeddings_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    print(
        f"Индекс построен: {len(records)} мемов, "
        f"размерность: {dim}, файлы: {index_path}, {meta_path}"
    )


def _load_index_and_meta() -> Tuple[faiss.IndexFlatIP, list[dict]]:
    cfg = load_app_config()
    emb_cfg = cfg.embeddings
    embeddings_dir = Path(emb_cfg.embeddings_dir)

    index_path = embeddings_dir / "index.faiss"
    meta_path = embeddings_dir / "meta.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "FAISS-индекс или метаданные не найдены. "
            "Сначала запустите build_faiss_index()."
        )

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text("utf-8"))
    return index, meta


def search_similar_by_image(
    image_path: Path, top_k: int = 5
) -> List[Tuple[float, dict]]:
    """
    Ищет похожие мемы в базе канала по картинке-кандидату.
    Возвращает список (similarity, meta_dict).
    """
    cfg = load_app_config()
    emb_cfg = cfg.embeddings

    embedder = MemeEmbedder(emb_cfg.image_model_name)
    index, meta = _load_index_and_meta()

    img = Image.open(image_path).convert("RGB")
    query_emb = embedder.encode_image(img)
    query_emb = np.expand_dims(query_emb, axis=0)

    scores, idxs = index.search(query_emb, top_k)
    scores = scores[0]
    idxs = idxs[0]

    results: List[Tuple[float, dict]] = []
    for score, idx in zip(scores, idxs, strict=False):
        if idx < 0:
            continue
        results.append((float(score), meta[idx]))
    return results


if __name__ == "__main__":
    build_faiss_index()


