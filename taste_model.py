"""
Модуль для работы с мультимодальной моделью "вкуса" мемов.
Использует SigLIP для изображений, SBERT для текста и обученный вектор вкуса.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

# Игнорируем предупреждения MPS (Apple Silicon)
warnings.filterwarnings("ignore", category=UserWarning)

# Настройки модели
TASTE_VECTOR_FILE = Path("mean_taste_multimodal.npy")
IMG_MODEL_NAME = "google/siglip-base-patch16-224"
TEXT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.6  # Порог похожести по умолчанию


class TasteModel:
    """
    Модель для оценки "вкуса" мемов на основе мультимодальных эмбеддингов.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Инициализирует модель вкуса.
        
        Args:
            device: Устройство для вычислений ('mps', 'cuda', 'cpu').
                   Если None, выбирается автоматически.
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        print(f"🚀 TasteModel использует устройство: {self.device}")
        
        # Загрузка моделей
        print("📥 Загрузка моделей (SigLIP + SBERT)...")
        self.img_processor = AutoProcessor.from_pretrained(
            IMG_MODEL_NAME, use_fast=False
        )
        self.img_model = AutoModel.from_pretrained(IMG_MODEL_NAME).to(self.device)
        self.text_embedder = SentenceTransformer(TEXT_MODEL_NAME).to(self.device)
        
        # Загрузка вектора вкуса
        self.taste_vector: Optional[np.ndarray] = None
        self.load_taste_vector()
    
    def load_taste_vector(self) -> bool:
        """
        Загружает обученный вектор вкуса из файла.
        
        Returns:
            True если вектор загружен успешно, False иначе.
        """
        if not TASTE_VECTOR_FILE.exists():
            print(f"⚠️  Файл вектора вкуса {TASTE_VECTOR_FILE} не найден.")
            print("   Используйте функцию learn_taste() для обучения модели.")
            return False
        
        self.taste_vector = np.load(TASTE_VECTOR_FILE)
        print(f"✅ Вектор вкуса загружен из {TASTE_VECTOR_FILE}")
        print(f"   Размерность: {self.taste_vector.shape}")
        return True
    
    def get_multimodal_embedding(
        self, image: Image.Image, text_content: str = ""
    ) -> Optional[np.ndarray]:
        """
        Создаёт мультимодальный эмбеддинг на основе изображения и текста.
        
        Args:
            image: Изображение PIL
            text_content: Текст для анализа (подпись или OCR-текст)
        
        Returns:
            Нормализованный вектор эмбеддинга или None при ошибке
        """
        try:
            # 1. Визуальный вектор (SigLIP)
            inputs = self.img_processor(images=image, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                img_features = self.img_model.get_image_features(**inputs)
            img_vec = (
                img_features / img_features.norm(p=2, dim=-1, keepdim=True)
            ).cpu().numpy().flatten()
            
            # 2. Текстовый вектор (SBERT)
            if text_content:
                text_vec = self.text_embedder.encode(text_content)
                text_vec = text_vec / (np.linalg.norm(text_vec) + 1e-8)
            else:
                text_vec = np.zeros(384)  # Размерность SBERT модели
            
            # 3. Объединение векторов
            combined = np.concatenate([img_vec, text_vec])
            return combined / np.linalg.norm(combined)
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка создания эмбеддинга: {exc}")
            return None
    
    def calculate_similarity(
        self, embedding: np.ndarray, threshold: float = SIMILARITY_THRESHOLD
    ) -> tuple[float, bool]:
        """
        Вычисляет похожесть эмбеддинга на вектор вкуса.
        
        Args:
            embedding: Вектор эмбеддинга мема
            threshold: Порог похожести
        
        Returns:
            Кортеж (similarity_score, is_similar)
        """
        if self.taste_vector is None:
            raise ValueError(
                "Вектор вкуса не загружен. Используйте load_taste_vector()"
            )
        
        similarity = float(np.dot(embedding, self.taste_vector))
        is_similar = similarity >= threshold
        return similarity, is_similar
    
    def evaluate_meme(
        self,
        image: Image.Image,
        text_content: str = "",
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> tuple[Optional[np.ndarray], Optional[float], Optional[bool]]:
        """
        Полная оценка мема: создаёт эмбеддинг и вычисляет похожесть.
        
        Args:
            image: Изображение PIL
            text_content: Текст для анализа
            threshold: Порог похожести
        
        Returns:
            Кортеж (embedding, similarity_score, is_similar)
        """
        embedding = self.get_multimodal_embedding(image, text_content)
        if embedding is None:
            return None, None, None
        
        similarity, is_similar = self.calculate_similarity(embedding, threshold)
        return embedding, similarity, is_similar


# Глобальный экземпляр модели (ленивая инициализация)
_taste_model_instance: Optional[TasteModel] = None


def get_taste_model() -> TasteModel:
    """
    Возвращает глобальный экземпляр модели вкуса (singleton).
    """
    global _taste_model_instance
    if _taste_model_instance is None:
        _taste_model_instance = TasteModel()
    return _taste_model_instance

