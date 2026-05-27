"""
Pipeline для демо-вебинтерфейса по ВКР:
"Семантический анализ русскоязычных диалогов для распознавания намерений".

Содержит:
- ленивую загрузку Whisper (faster-whisper) для русского языка;
- intent-классификацию: предпочтительно single-task RuBERT runtime из ноутбука 11
  (best модель: macro-F1 0.7770, accuracy 0.9126 на test), при отсутствии
  артефактов — старая joblib-модель или rule-based fallback;
- тематический классификатор (sentence-transformers + центроиды кластеров
  или keyword fallback);
- заглушку суммаризации;
- общую функцию analyze(text) -> dict (JSON-совместимый).

Все модели опциональны — если артефакты не лежат в ./models/, будут
использованы mock/fallback ветки, чтобы webapp запустился на пустом VPS.
"""

from __future__ import annotations

import json
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
    # single-task RuBERT runtime (notebook 11)
    "torch_intent_runtime": None,        # SingleTaskIntentModelRuntime instance after lazy build
    "torch_intent_state_path": None,     # путь к .pt, обнаруженный при load_artifacts
    "torch_intent_config": None,         # dict из multitask_config.json (если есть)
    "torch_intent_load_error": None,     # последнее предупреждение при загрузке
    "torch_intent_attempted": False,     # уже пытались lazy-build
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

def _env_flag(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on", "y"}


def load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    """Загружает опциональные артефакты моделей.

    Никогда не падает — при отсутствии файлов соответствующие ключи остаются
    None, и downstream-функции переходят в fallback-режим.

    Сам тяжёлый PyTorch-encoder (`single_task_intent_model.pt`) здесь не
    инициализируется: на этом этапе мы только обнаруживаем файлы и читаем
    `intent_label_encoder.joblib` + `multitask_config.json`. Реальная сборка
    модели (`AutoModel.from_pretrained` + `load_state_dict`) выполняется
    лениво при первом обращении к `predict_intent`, чтобы старт UI оставался
    быстрым.
    """
    models_path = Path(models_dir)
    _ARTIFACTS["models_dir"] = str(models_path.resolve())

    # intent label encoder (используется и старым joblib-pipeline, и новым torch runtime)
    intent_label_path = models_path / "intent_label_encoder.joblib"
    if intent_label_path.exists():
        try:
            import joblib  # noqa: WPS433
            _ARTIFACTS["intent_label_encoder"] = joblib.load(intent_label_path)
            logger.info("Загружен intent_label_encoder.joblib")
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось загрузить intent_label_encoder.joblib: %s", exc)

    # старый sklearn-style intent_model (оставлен для обратной совместимости)
    intent_model_path = models_path / "intent_model.joblib"
    if intent_model_path.exists():
        try:
            import joblib  # noqa: WPS433
            _ARTIFACTS["intent_model"] = joblib.load(intent_model_path)
            logger.info("Загружен intent_model.joblib")
        except Exception as exc:  # pragma: no cover - зависит от артефактов
            logger.warning("Не удалось загрузить intent_model.joblib: %s", exc)

    # multitask config из ноутбука 11
    config_path = models_path / "multitask_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _ARTIFACTS["torch_intent_config"] = json.load(f)
            logger.info(
                "Загружен multitask_config.json: model_name=%s, num_intents=%s",
                _ARTIFACTS["torch_intent_config"].get("model_name"),
                _ARTIFACTS["torch_intent_config"].get("num_intents"),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось прочитать multitask_config.json: %s", exc)

    # single-task RuBERT state_dict (notebook 11, best модель)
    intent_state_file = os.environ.get("INTENT_MODEL_FILE", "single_task_intent_model.pt")
    intent_state_path = models_path / intent_state_file
    if intent_state_path.exists():
        _ARTIFACTS["torch_intent_state_path"] = str(intent_state_path.resolve())
        logger.info(
            "Обнаружен single-task RuBERT state_dict: %s (lazy-load при первом predict_intent)",
            intent_state_path.name,
        )

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


def get_artifact_status() -> Dict[str, Any]:
    """Возвращает статус артефактов для отображения в UI / диагностики."""
    _ensure_loaded()
    has_torch_state = _ARTIFACTS.get("torch_intent_state_path") is not None
    has_encoder = _ARTIFACTS.get("intent_label_encoder") is not None
    has_config = _ARTIFACTS.get("torch_intent_config") is not None
    runtime_loaded = _ARTIFACTS.get("torch_intent_runtime") is not None
    enabled = _env_flag("ENABLE_TORCH_INTENT_MODEL", True)

    if enabled and has_torch_state and has_encoder:
        intent_mode = "single_task_rubert_model"
    elif _ARTIFACTS.get("intent_model") is not None:
        intent_mode = "sklearn_intent_model"
    else:
        intent_mode = "rule_based_fallback"

    return {
        "models_dir": _ARTIFACTS.get("models_dir"),
        "torch_intent_enabled": enabled,
        "torch_intent_state_found": has_torch_state,
        "torch_intent_state_path": _ARTIFACTS.get("torch_intent_state_path"),
        "intent_label_encoder_found": has_encoder,
        "multitask_config_found": has_config,
        "torch_intent_runtime_loaded": runtime_loaded,
        "torch_intent_load_error": _ARTIFACTS.get("torch_intent_load_error"),
        "intent_mode_planned": intent_mode,
        "topic_centroids_found": _ARTIFACTS.get("topic_centroids") is not None,
        "topic_metadata_found": _ARTIFACTS.get("topic_metadata") is not None,
    }


def _ensure_loaded() -> None:
    if not _ARTIFACTS["loaded"]:
        load_artifacts(os.environ.get("MODELS_DIR", "models"))


# ---------------------------------------------------------------------------
# Single-task RuBERT runtime (notebook 11)
# ---------------------------------------------------------------------------

_DEFAULT_TORCH_MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
_DEFAULT_TORCH_MAX_LEN = 128
_DEFAULT_TORCH_DROPOUT = 0.1


class SingleTaskIntentModelRuntime:
    """Runtime-обёртка `SingleTaskIntentModel` из ноутбука 11.

    Архитектура повторяет `SingleTaskIntentModel`:
        encoder = AutoModel.from_pretrained(base_model_name)
        shared/proj = Linear(h, h) -> GELU -> LayerNorm -> Dropout
        intent_head = Linear(h, num_intents)
    Pooling — mean по attention mask.

    Имя поля проекции в state_dict из ноутбука — `proj.*` (см. cell 5
    notebook 11). На случай чужих чекпойнтов поддерживается
    `load_state_dict(strict=False)` с предупреждением.
    """

    def __init__(
        self,
        base_model_name: str,
        num_intents: int,
        dropout: float = _DEFAULT_TORCH_DROPOUT,
        max_len: int = _DEFAULT_TORCH_MAX_LEN,
        device: Optional[str] = None,
    ):
        import torch  # noqa: WPS433
        from torch import nn  # noqa: WPS433
        from transformers import AutoModel, AutoTokenizer  # noqa: WPS433

        self._torch = torch
        self.base_model_name = base_model_name
        self.num_intents = int(num_intents)
        self.max_len = int(max_len)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.intent_head = nn.Linear(hidden, self.num_intents)

        # Собираем nn.Module-обёртку, чтобы load_state_dict видел все веса
        # под теми же именами, что сохраняет notebook 11.
        class _Wrap(nn.Module):
            def __init__(wself):
                super().__init__()
                wself.encoder = self.encoder
                wself.proj = self.proj
                wself.intent_head = self.intent_head

        self._module = _Wrap()

    def load_state_dict_from_path(self, path: str) -> Dict[str, Any]:
        torch = self._torch
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and not any(
            k.startswith(("encoder.", "proj.", "intent_head.")) for k in state
        ):
            # на случай, если кто-то завернул в {"state_dict": ...}
            state = state["state_dict"]

        # Совместимость: ноутбук сохраняет ключи через model.state_dict() без
        # дополнительных префиксов, поэтому имена должны совпадать. Но если
        # state_dict содержит головы из multi-task модели (topic_head, ...) —
        # отфильтруем их, чтобы strict=False не молчал на shape mismatch.
        keep_prefixes = ("encoder.", "proj.", "intent_head.")
        filtered = {k: v for k, v in state.items() if k.startswith(keep_prefixes)}
        if not filtered:
            raise RuntimeError(
                "В state_dict не найдены ключи encoder./proj./intent_head. "
                "Файл несовместим с SingleTaskIntentModelRuntime."
            )

        try:
            missing, unexpected = self._module.load_state_dict(filtered, strict=True)
            warn = None
        except Exception as exc_strict:
            # fallback: strict=False
            result = self._module.load_state_dict(filtered, strict=False)
            missing = list(getattr(result, "missing_keys", []) or [])
            unexpected = list(getattr(result, "unexpected_keys", []) or [])
            warn = f"strict=True не прошёл ({exc_strict}); загрузка через strict=False"
            logger.warning("SingleTask state_dict: %s", warn)

        self._module.to(self.device)
        self._module.eval()
        return {
            "missing_keys": list(missing) if missing else [],
            "unexpected_keys": list(unexpected) if unexpected else [],
            "warning": warn,
            "device": self.device,
        }

    def predict(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        torch = self._torch
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            proj = self.proj(pooled)
            logits = self.intent_head(proj)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        topk_idx = np.argsort(probs)[::-1][: max(1, int(top_k))]
        return {
            "argmax_index": idx,
            "argmax_confidence": float(probs[idx]),
            "probs": probs.astype(float).tolist(),
            "topk": [
                {"index": int(j), "confidence": float(probs[j])} for j in topk_idx
            ],
        }


def _build_torch_intent_runtime() -> Optional[SingleTaskIntentModelRuntime]:
    """Ленивая инициализация single-task RuBERT runtime.

    Возвращает None и записывает причину в `torch_intent_load_error`, если
    что-то пошло не так. Никогда не пробрасывает исключения.
    """
    if _ARTIFACTS.get("torch_intent_runtime") is not None:
        return _ARTIFACTS["torch_intent_runtime"]

    if not _env_flag("ENABLE_TORCH_INTENT_MODEL", True):
        _ARTIFACTS["torch_intent_load_error"] = "ENABLE_TORCH_INTENT_MODEL=false"
        return None

    state_path = _ARTIFACTS.get("torch_intent_state_path")
    label_encoder = _ARTIFACTS.get("intent_label_encoder")
    if not state_path:
        _ARTIFACTS["torch_intent_load_error"] = "state_dict не найден"
        return None
    if label_encoder is None or not hasattr(label_encoder, "classes_"):
        _ARTIFACTS["torch_intent_load_error"] = "intent_label_encoder.joblib не найден"
        return None

    config = _ARTIFACTS.get("torch_intent_config") or {}
    base_model_name = (
        os.environ.get("INTENT_ENCODER_NAME")
        or config.get("model_name")
        or _DEFAULT_TORCH_MODEL_NAME
    )
    max_len = int(config.get("max_len") or _DEFAULT_TORCH_MAX_LEN)
    num_intents_cfg = config.get("num_intents")
    num_intents = int(num_intents_cfg) if num_intents_cfg else len(label_encoder.classes_)
    if num_intents != len(label_encoder.classes_):
        logger.warning(
            "num_intents из config (%s) != len(label_encoder.classes_) (%s); используем encoder",
            num_intents_cfg, len(label_encoder.classes_),
        )
        num_intents = len(label_encoder.classes_)

    try:
        runtime = SingleTaskIntentModelRuntime(
            base_model_name=base_model_name,
            num_intents=num_intents,
            dropout=_DEFAULT_TORCH_DROPOUT,
            max_len=max_len,
        )
        info = runtime.load_state_dict_from_path(state_path)
    except Exception as exc:
        _ARTIFACTS["torch_intent_load_error"] = f"build/load failed: {exc}"
        logger.warning("Single-task RuBERT runtime не загружен: %s", exc)
        return None

    _ARTIFACTS["torch_intent_runtime"] = runtime
    if info.get("warning"):
        _ARTIFACTS["torch_intent_load_error"] = info["warning"]
    else:
        _ARTIFACTS["torch_intent_load_error"] = None
    logger.info(
        "Single-task RuBERT runtime готов: base=%s, num_intents=%d, device=%s",
        base_model_name, num_intents, info.get("device"),
    )
    return runtime


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
    """Предсказывает intent.

    Порядок: single-task RuBERT runtime (notebook 11) → старый
    sklearn joblib intent_model → rule-based fallback.
    """
    _ensure_loaded()
    if not text or not text.strip():
        return {"label": "other", "confidence": 0.0, "mode": "empty_input"}

    encoder = _ARTIFACTS.get("intent_label_encoder")

    # 1) Single-task RuBERT runtime (best модель из ноутбука 11)
    if (
        _env_flag("ENABLE_TORCH_INTENT_MODEL", True)
        and _ARTIFACTS.get("torch_intent_state_path")
        and encoder is not None
        and hasattr(encoder, "classes_")
    ):
        if not _ARTIFACTS["torch_intent_attempted"]:
            _ARTIFACTS["torch_intent_attempted"] = True
            _build_torch_intent_runtime()

        runtime = _ARTIFACTS.get("torch_intent_runtime")
        if runtime is not None:
            try:
                pred = runtime.predict(text, top_k=5)
                idx = int(pred["argmax_index"])
                classes = list(encoder.classes_)
                label = str(classes[idx]) if 0 <= idx < len(classes) else INTENT_CLASSES[idx % len(INTENT_CLASSES)]
                topk = [
                    {
                        "label": str(classes[int(item["index"])])
                        if 0 <= int(item["index"]) < len(classes)
                        else f"class_{item['index']}",
                        "confidence": float(item["confidence"]),
                    }
                    for item in pred.get("topk", [])
                ]
                return {
                    "label": label,
                    "confidence": float(pred["argmax_confidence"]),
                    "mode": "single_task_rubert_model",
                    "topk": topk,
                }
            except Exception as exc:
                logger.warning("Torch single-task runtime упал, fallback. %s", exc)
                _ARTIFACTS["torch_intent_load_error"] = f"predict failed: {exc}"

    # 2) Старая sklearn-модель (обратная совместимость)
    model = _ARTIFACTS.get("intent_model")
    if model is not None:
        try:
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
                "mode": "sklearn_intent_model",
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
            "intent_topk": None,
            "topic_cluster_id": None,
            "topic_cluster_name": None,
            "topic_cluster_description": None,
            "topic_top_words": [],
            "topic_confidence": None,
            "topic_mode": "empty_input",
            "summary": None,
            "summary_status": "Пустой ввод — анализ не выполнен",
            "artifact_status": get_artifact_status(),
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
        "intent_topk": intent.get("topk"),
        "topic_cluster_id": topic.get("cluster_id"),
        "topic_cluster_name": topic.get("name"),
        "topic_cluster_description": topic.get("description"),
        "topic_top_words": topic.get("top_words", []),
        "topic_confidence": topic.get("confidence"),
        "topic_mode": topic.get("mode"),
        "summary": summary,
        "summary_status": status,
        "artifact_status": get_artifact_status(),
    }


if __name__ == "__main__":
    # Простой smoke-test
    import json
    load_artifacts(os.environ.get("MODELS_DIR", "models"))
    sample = "Здравствуйте, я хочу забронировать билет на концерт в Москве"
    print(json.dumps(analyze(sample), ensure_ascii=False, indent=2))
