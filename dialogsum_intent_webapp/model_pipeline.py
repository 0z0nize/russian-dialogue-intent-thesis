"""
Pipeline для демо-вебинтерфейса по ВКР:
"Семантический анализ русскоязычных диалогов для распознавания намерений".

Содержит:
- ленивую загрузку Whisper (faster-whisper) для русского языка;
- intent-классификацию (joblib-модель из ВКР или rule-based fallback);
- тематический классификатор (sentence-transformers + центроиды кластеров
  или keyword fallback);
- заглушку суммаризации;
- общую функцию analyze(text) -> dict (JSON-совместимый).

Все модели опциональны — если артефакты не лежат в ./models/, будут
использованы mock/fallback ветки, чтобы webapp запустился на пустом VPS.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

# ---------------------------------------------------------------------------
# Глобальное состояние артефактов и моделей
# ---------------------------------------------------------------------------

_ARTIFACTS: Dict[str, Any] = {
    "intent_model": None,
    "intent_label_encoder": None,
    "topic_centroids": None,
    "topic_metadata": None,
    "sentence_encoder": None,
    "loaded": False,
    "models_dir": None,
}

_WHISPER_MODEL = None  # ленивая загрузка


# ---------------------------------------------------------------------------
# Класс intents (из ВКР, 14 классов)
# ---------------------------------------------------------------------------

INTENT_CLASSES: List[str] = [
    "greeting",
    "thanks",
    "farewell",
    "informational_request",
    "service_request",
    "purchase_or_booking_request",
    "complaint",
    "problem_report",
    "arrangement",
    "confirmation",
    "rejection",
    "suggestion_or_recommendation",
    "opinion_or_preference",
    "other",
]

# Ключевые слова для rule-based fallback по интентам.
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "greeting": ["здравствуй", "привет", "добрый день", "доброе утро", "добрый вечер", "здравствуйте"],
    "thanks": ["спасибо", "благодар", "признателен", "благодарю"],
    "farewell": ["до свидания", "пока", "всего доброго", "до встречи", "счастливо"],
    "informational_request": ["как узнать", "подскажите", "сколько стоит", "какие есть", "что такое", "почему", "когда", "где"],
    "service_request": ["помогите", "нужна помощь", "проконсультируйте", "оформите", "сделайте", "нужно оформить"],
    "purchase_or_booking_request": ["забронировать", "купить", "заказать", "оплатить", "оформить заказ", "хочу купить", "бронь"],
    "complaint": ["жалоба", "недовольн", "ужасно", "плохо", "возмутитель", "не работает", "обманули"],
    "problem_report": ["проблема", "не получается", "ошибка", "сломал", "не могу", "не работает", "сбой"],
    "arrangement": ["договоримся", "встретимся", "давайте в", "назначим", "когда удобно", "запланируем"],
    "confirmation": ["да", "согласен", "хорошо", "подтвержд", "конечно", "договорились"],
    "rejection": ["нет", "отказ", "не нужно", "не хочу", "не буду", "не подходит"],
    "suggestion_or_recommendation": ["рекомендую", "советую", "предлагаю", "попробуйте", "лучше выбрать"],
    "opinion_or_preference": ["мне нравится", "я думаю", "по-моему", "считаю", "предпочитаю", "люблю"],
}


# ---------------------------------------------------------------------------
# Темы для keyword fallback (отражают темы из ВКР)
# ---------------------------------------------------------------------------

_TOPIC_FALLBACK: List[Dict[str, Any]] = [
    {
        "id": 0,
        "name": "развлечения",
        "description": "Кино, сериалы, игры, досуг и развлекательные мероприятия.",
        "top_words": ["фильм", "сериал", "игра", "кино", "развлеч", "досуг"],
    },
    {
        "id": 1,
        "name": "музыкальные события",
        "description": "Концерты, музыкальные фестивали, билеты, исполнители.",
        "top_words": ["концерт", "музык", "билет", "группа", "фестивал", "певец"],
    },
    {
        "id": 2,
        "name": "дом",
        "description": "Бытовые темы, домашние дела, аренда и обустройство жилья.",
        "top_words": ["дом", "квартир", "аренд", "уборк", "кухн", "ремонт быт"],
    },
    {
        "id": 3,
        "name": "жалоба",
        "description": "Жалобы на сервис, продукт или ситуацию.",
        "top_words": ["жалоб", "недоволен", "плох", "обман", "верн деньги"],
    },
    {
        "id": 4,
        "name": "образование",
        "description": "Учёба, курсы, экзамены, университет.",
        "top_words": ["учеб", "курс", "экзамен", "универ", "школ", "лекц"],
    },
    {
        "id": 5,
        "name": "поиск книг",
        "description": "Поиск, покупка и обсуждение книг.",
        "top_words": ["книг", "автор", "читать", "роман", "издани", "библиотек"],
    },
    {
        "id": 6,
        "name": "путешествия",
        "description": "Поездки, билеты, отели, маршруты, бронирование путешествий.",
        "top_words": ["путешеств", "отель", "билет", "поездк", "тур", "виза", "брон"],
    },
    {
        "id": 7,
        "name": "работа",
        "description": "Профессиональные задачи, коллеги, рабочие процессы.",
        "top_words": ["работ", "офис", "проект", "коллег", "задач", "начальник"],
    },
    {
        "id": 8,
        "name": "собеседование",
        "description": "Поиск работы, резюме, собеседования, hr-вопросы.",
        "top_words": ["собеседован", "вакансия", "резюме", "hr", "оффер", "найм"],
    },
    {
        "id": 9,
        "name": "ремонт/обслуживание",
        "description": "Ремонт техники, обслуживание автомобиля, сервисные услуги.",
        "top_words": ["ремонт", "обслуживан", "сервис", "почин", "мастер", "техник"],
    },
]


# ---------------------------------------------------------------------------
# Загрузка артефактов
# ---------------------------------------------------------------------------

def load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    """Загружает опциональные артефакты моделей.

    Никогда не падает — при отсутствии файлов соответствующие ключи остаются
    None, и downstream-функции переходят в fallback-режим.
    """
    models_path = Path(models_dir)
    _ARTIFACTS["models_dir"] = str(models_path.resolve())

    # intent model + label encoder
    intent_model_path = models_path / "intent_model.joblib"
    intent_label_path = models_path / "intent_label_encoder.joblib"
    if intent_model_path.exists():
        try:
            import joblib  # noqa: WPS433
            _ARTIFACTS["intent_model"] = joblib.load(intent_model_path)
            logger.info("Загружен intent_model.joblib")
        except Exception as exc:  # pragma: no cover - зависит от артефактов
            logger.warning("Не удалось загрузить intent_model.joblib: %s", exc)
    if intent_label_path.exists():
        try:
            import joblib  # noqa: WPS433
            _ARTIFACTS["intent_label_encoder"] = joblib.load(intent_label_path)
            logger.info("Загружен intent_label_encoder.joblib")
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось загрузить intent_label_encoder.joblib: %s", exc)

    # topic centroids
    centroids_path = models_path / "topic_centroids.npy"
    if centroids_path.exists():
        try:
            _ARTIFACTS["topic_centroids"] = np.load(centroids_path)
            logger.info("Загружен topic_centroids.npy: shape=%s", _ARTIFACTS["topic_centroids"].shape)
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось загрузить topic_centroids.npy: %s", exc)

    # topic metadata
    meta_path = models_path / "topic_metadata.parquet"
    if meta_path.exists():
        try:
            import pandas as pd  # noqa: WPS433
            _ARTIFACTS["topic_metadata"] = pd.read_parquet(meta_path)
            logger.info("Загружен topic_metadata.parquet: rows=%d", len(_ARTIFACTS["topic_metadata"]))
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось загрузить topic_metadata.parquet: %s", exc)

    _ARTIFACTS["loaded"] = True
    return _ARTIFACTS


def _ensure_loaded() -> None:
    if not _ARTIFACTS["loaded"]:
        load_artifacts(os.environ.get("MODELS_DIR", "models"))


# ---------------------------------------------------------------------------
# Sentence encoder (для тематики, если есть centroids)
# ---------------------------------------------------------------------------

def _get_sentence_encoder():
    """Ленивая инициализация sentence-transformers (только если нужно)."""
    if _ARTIFACTS["sentence_encoder"] is not None:
        return _ARTIFACTS["sentence_encoder"]
    try:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
        model_name = os.environ.get("SENTENCE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("Инициализация sentence-transformers: %s", model_name)
        _ARTIFACTS["sentence_encoder"] = SentenceTransformer(model_name)
        return _ARTIFACTS["sentence_encoder"]
    except Exception as exc:  # pragma: no cover - зависит от окружения
        logger.warning("sentence-transformers недоступен: %s", exc)
        return None


# ---------------------------------------------------------------------------
# STT: faster-whisper (ленивая загрузка)
# ---------------------------------------------------------------------------

def _get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - faster-whisper может отсутствовать
        logger.error("faster-whisper не установлен: %s", exc)
        return None

    model_size = os.environ.get("WHISPER_MODEL_SIZE", "medium")

    # Определяем устройство
    device = "cpu"
    compute_type = "int8"
    try:
        import torch  # noqa: WPS433
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
    except Exception:  # pragma: no cover
        pass

    logger.info("Загрузка Whisper: size=%s, device=%s, compute_type=%s", model_size, device, compute_type)
    try:
        _WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:  # pragma: no cover
        logger.error("Не удалось инициализировать Whisper: %s", exc)
        _WHISPER_MODEL = None
    return _WHISPER_MODEL


def transcribe_audio(audio_path: str) -> str:
    """Распознаёт русскую речь в аудиофайле и возвращает текст.

    Возвращает пустую строку, если файл некорректен или Whisper недоступен.
    """
    if not audio_path:
        return ""
    if not Path(audio_path).exists():
        logger.warning("Аудиофайл не найден: %s", audio_path)
        return ""

    model = _get_whisper()
    if model is None:
        return "[STT недоступен: faster-whisper не загружен]"

    try:
        segments, _info = model.transcribe(audio_path, language="ru", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
    except Exception as exc:  # pragma: no cover - runtime
        logger.exception("Ошибка распознавания: %s", exc)
        return f"[Ошибка распознавания: {exc}]"


# ---------------------------------------------------------------------------
# Intent prediction
# ---------------------------------------------------------------------------

def _intent_rule_based(text: str) -> Dict[str, Any]:
    text_low = text.lower()
    scores: Dict[str, int] = {}
    for label, kws in _INTENT_KEYWORDS.items():
        hit = sum(1 for kw in kws if kw in text_low)
        if hit:
            scores[label] = hit
    if not scores:
        return {
            "label": "other",
            "confidence": 0.2,
            "mode": "rule_based_fallback",
        }
    label, hits = max(scores.items(), key=lambda kv: kv[1])
    total = sum(scores.values())
    confidence = min(0.95, 0.5 + 0.1 * hits)
    return {
        "label": label,
        "confidence": float(confidence),
        "mode": "rule_based_fallback",
        "raw_scores": {k: v / total for k, v in scores.items()},
    }


def predict_intent(text: str) -> Dict[str, Any]:
    """Предсказывает intent. Использует joblib-модель из ВКР, если она есть."""
    _ensure_loaded()
    if not text or not text.strip():
        return {"label": "other", "confidence": 0.0, "mode": "empty_input"}

    model = _ARTIFACTS.get("intent_model")
    encoder = _ARTIFACTS.get("intent_label_encoder")

    if model is not None:
        try:
            # Пытаемся как pipeline (vectorizer внутри)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([text])[0]
                idx = int(np.argmax(proba))
                confidence = float(proba[idx])
            else:
                idx = int(model.predict([text])[0])
                confidence = 0.5
            if encoder is not None and hasattr(encoder, "classes_"):
                label = str(encoder.classes_[idx])
            elif hasattr(model, "classes_"):
                label = str(model.classes_[idx])
            else:
                label = INTENT_CLASSES[idx % len(INTENT_CLASSES)]
            return {
                "label": label,
                "confidence": confidence,
                "mode": "model",
            }
        except Exception as exc:
            logger.warning("Intent модель не сработала, fallback. %s", exc)

    return _intent_rule_based(text)


# ---------------------------------------------------------------------------
# Topic prediction
# ---------------------------------------------------------------------------

def _topic_rule_based(text: str) -> Dict[str, Any]:
    text_low = text.lower()
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    for topic in _TOPIC_FALLBACK:
        score = sum(1 for w in topic["top_words"] if w in text_low)
        if score > best_score:
            best_score = score
            best = topic
    if best is None:
        return {
            "cluster_id": -1,
            "name": "не определена",
            "description": "Тема не определена по ключевым словам.",
            "top_words": [],
            "confidence": 0.0,
            "mode": "rule_based_fallback",
        }
    confidence = min(0.9, 0.4 + 0.1 * best_score)
    return {
        "cluster_id": best["id"],
        "name": best["name"],
        "description": best["description"],
        "top_words": best["top_words"],
        "confidence": float(confidence),
        "mode": "rule_based_fallback",
    }


def _topic_meta_row(cluster_id: int) -> Optional[Dict[str, Any]]:
    meta = _ARTIFACTS.get("topic_metadata")
    if meta is None:
        return None
    try:
        # Ожидаем колонки: cluster_id, name, description, top_words (list/str)
        if "cluster_id" in meta.columns:
            row = meta[meta["cluster_id"] == cluster_id]
        else:
            row = meta.iloc[[cluster_id]]
        if row.empty:
            return None
        rec = row.iloc[0].to_dict()
        top_words = rec.get("top_words", [])
        if isinstance(top_words, str):
            top_words = [w.strip() for w in re.split(r"[;,]", top_words) if w.strip()]
        return {
            "cluster_id": int(rec.get("cluster_id", cluster_id)),
            "name": str(rec.get("name", f"cluster_{cluster_id}")),
            "description": str(rec.get("description", "")),
            "top_words": list(top_words),
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("topic_metadata чтение упало: %s", exc)
        return None


def predict_topic(text: str) -> Dict[str, Any]:
    """Предсказывает тематический кластер.

    Если есть centroids + sentence-transformers — берём ближайший кластер
    по косинусной близости. Иначе — keyword fallback из ВКР-тем.
    """
    _ensure_loaded()
    if not text or not text.strip():
        return {
            "cluster_id": -1,
            "name": "не определена",
            "description": "Пустой ввод.",
            "top_words": [],
            "confidence": 0.0,
            "mode": "empty_input",
        }

    centroids = _ARTIFACTS.get("topic_centroids")
    if centroids is not None:
        encoder = _get_sentence_encoder()
        if encoder is not None:
            try:
                emb = encoder.encode([text], normalize_embeddings=True)[0]
                cents = centroids
                # Нормализуем центроиды на всякий случай
                norms = np.linalg.norm(cents, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                cents_n = cents / norms
                sims = cents_n @ emb
                idx = int(np.argmax(sims))
                confidence = float((sims[idx] + 1.0) / 2.0)  # [-1,1] -> [0,1]
                meta = _topic_meta_row(idx) or {
                    "cluster_id": idx,
                    "name": f"cluster_{idx}",
                    "description": "Метаданные кластера недоступны.",
                    "top_words": [],
                }
                meta.update({"confidence": confidence, "mode": "model"})
                return meta
            except Exception as exc:
                logger.warning("Topic model не сработал, fallback. %s", exc)

    return _topic_rule_based(text)


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize(text: str):
    """Заглушка суммаризации. Возвращает кортеж (summary, status)."""
    if not text or not text.strip():
        return None, "Суммаризация не подключена в демо-версии"
    return None, "Суммаризация не подключена в демо-версии"


# ---------------------------------------------------------------------------
# Главная функция analyze
# ---------------------------------------------------------------------------

def analyze(text: str) -> Dict[str, Any]:
    """Полный анализ текста: intent + topic + summary."""
    text = (text or "").strip()
    if not text:
        return {
            "input_text": "",
            "intent_label": None,
            "intent_confidence": None,
            "intent_mode": "empty_input",
            "topic_cluster_id": None,
            "topic_cluster_name": None,
            "topic_cluster_description": None,
            "topic_top_words": [],
            "topic_confidence": None,
            "topic_mode": "empty_input",
            "summary": None,
            "summary_status": "Пустой ввод — анализ не выполнен",
        }

    try:
        intent = predict_intent(text)
    except Exception as exc:
        logger.exception("predict_intent ошибка: %s", exc)
        intent = {"label": "other", "confidence": 0.0, "mode": f"error: {exc}"}

    try:
        topic = predict_topic(text)
    except Exception as exc:
        logger.exception("predict_topic ошибка: %s", exc)
        topic = {
            "cluster_id": -1,
            "name": "не определена",
            "description": str(exc),
            "top_words": [],
            "confidence": 0.0,
            "mode": f"error: {exc}",
        }

    summary, status = summarize(text)

    return {
        "input_text": text,
        "intent_label": intent.get("label"),
        "intent_confidence": intent.get("confidence"),
        "intent_mode": intent.get("mode"),
        "topic_cluster_id": topic.get("cluster_id"),
        "topic_cluster_name": topic.get("name"),
        "topic_cluster_description": topic.get("description"),
        "topic_top_words": topic.get("top_words", []),
        "topic_confidence": topic.get("confidence"),
        "topic_mode": topic.get("mode"),
        "summary": summary,
        "summary_status": status,
    }


if __name__ == "__main__":
    # Простой smoke-test
    import json
    load_artifacts(os.environ.get("MODELS_DIR", "models"))
    sample = "Здравствуйте, я хочу забронировать билет на концерт в Москве"
    print(json.dumps(analyze(sample), ensure_ascii=False, indent=2))
