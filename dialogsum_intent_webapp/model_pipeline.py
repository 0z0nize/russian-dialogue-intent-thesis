"""
Pipeline для демо-вебинтерфейса по ВКР:
"Семантический анализ русскоязычных диалогов для распознавания намерений".

Содержит:
- ленивую загрузку Whisper (faster-whisper) для русского языка;
- intent-классификацию: предпочтительно single-task RuBERT runtime
  (best модель: macro-F1 0.7770, accuracy 0.9126 на test) из артефактов
  проекта; при отсутствии — старая joblib-модель или rule-based fallback;
- тематический классификатор (sentence-transformers + центроиды кластеров
  или keyword fallback);
- реальную суммаризацию через HuggingFace Transformers
  (по умолчанию IlyaGusev/rut5_base_sum_gazeta);
- общую функцию analyze(text) -> dict (JSON-совместимый).

Артефакты ищутся сначала в локальной папке `models/`, а если файлов нет —
скачиваются с Hugging Face Hub (по умолчанию из репозитория
`ozonize/dialogsum-ru-intent-rubert`). Все модели опциональны: если
артефакты не найдены и Hub недоступен, включаются rule-based ветки, чтобы
webapp запустился даже на пустом VPS.
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
    # single-task RuBERT runtime
    "torch_intent_runtime": None,        # SingleTaskIntentModelRuntime instance after lazy build
    "torch_intent_state_path": None,     # путь к .pt, обнаруженный при load_artifacts
    "torch_intent_state_source": None,   # "local" | "huggingface_hub"
    "torch_intent_config": None,         # dict из multitask_config.json (если есть)
    "torch_intent_config_source": None,
    "torch_intent_label_encoder_source": None,
    "torch_intent_load_error": None,     # последнее предупреждение при загрузке
    "torch_intent_attempted": False,     # уже пытались lazy-build
    # summarization
    "summarizer_model": None,
    "summarizer_tokenizer": None,
    "summarizer_attempted": False,
    "summarizer_source": None,           # "local" | "huggingface_hub"
    "summarizer_path_or_name": None,
    "summarizer_load_error": None,
    "summarizer_device": None,
    "loaded": False,
    "models_dir": None,
    "hf_repo_id": None,
    "hf_available": False,
    "hf_download_error": None,
}

_WHISPER_MODEL = None  # ленивая загрузка


# ---------------------------------------------------------------------------
# Hugging Face Hub
# ---------------------------------------------------------------------------

HF_INTENT_REPO_ID_DEFAULT = "ozonize/dialogsum-ru-intent-rubert"

# Имена артефактов, которые мы умеем тянуть с Hub при необходимости.
_HF_INTENT_FILES = (
    "single_task_intent_model.pt",
    "intent_label_encoder.joblib",
    "multitask_config.json",
    "multitask_intent_topic_model.pt",
    "topic_label_encoder.joblib",
    "coarse_topic_label_encoder.joblib",
)


def _hf_repo_id() -> str:
    return os.environ.get("HF_INTENT_REPO_ID", HF_INTENT_REPO_ID_DEFAULT)


def _hf_token() -> Optional[str]:
    """Опциональный токен (для приватных репозиториев)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _hf_download(filename: str) -> Optional[Path]:
    """Скачивает один файл из Hub-репозитория интентов в локальный кэш.

    Возвращает Path к локальному файлу или None при ошибке. Все ошибки
    подавляются и пишутся в логи — приложение должно работать и без сети.
    """
    if not _env_flag("ENABLE_HF_DOWNLOAD", True):
        return None
    try:
        from huggingface_hub import hf_hub_download  # noqa: WPS433
    except Exception as exc:
        _ARTIFACTS["hf_download_error"] = f"huggingface_hub не установлен: {exc}"
        logger.warning("HF Hub: %s", _ARTIFACTS["hf_download_error"])
        return None

    repo_id = _hf_repo_id()
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=_hf_token(),
        )
        logger.info("HF Hub: скачан %s из %s -> %s", filename, repo_id, local)
        return Path(local)
    except Exception as exc:
        _ARTIFACTS["hf_download_error"] = f"{filename}: {exc}"
        logger.warning("HF Hub: не удалось скачать %s из %s: %s", filename, repo_id, exc)
        return None


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


def _resolve_artifact_local(filename: str, models_path: Path) -> Optional[Path]:
    """Возвращает локальный путь к артефакту, если он существует."""
    try:
        candidate = models_path / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    except OSError:
        return None
    return None


def _resolve_artifact(filename: str, models_path: Path) -> Optional[Path]:
    """Локально, иначе — пробуем скачать с HF Hub."""
    local = _resolve_artifact_local(filename, models_path)
    if local is not None:
        return local
    return _hf_download(filename)


def load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    """Загружает опциональные артефакты моделей.

    Никогда не падает — при отсутствии файлов соответствующие ключи остаются
    None, и downstream-функции переходят в fallback-режим.

    Порядок поиска для intent-артефактов:
      1. локальная папка `models_dir`;
      2. Hugging Face Hub (`$HF_INTENT_REPO_ID`, по умолчанию
         `ozonize/dialogsum-ru-intent-rubert`).

    Сам тяжёлый PyTorch-encoder (`single_task_intent_model.pt`) здесь не
    инициализируется: на этом этапе мы только обнаруживаем файлы и читаем
    `intent_label_encoder.joblib` + `multitask_config.json`. Реальная сборка
    модели (`AutoModel.from_pretrained` + `load_state_dict`) выполняется
    лениво при первом обращении к `predict_intent`, чтобы старт UI оставался
    быстрым.
    """
    models_path = Path(models_dir)
    _ARTIFACTS["models_dir"] = str(models_path.resolve()) if models_path.exists() else str(models_path)
    _ARTIFACTS["hf_repo_id"] = _hf_repo_id()
    _ARTIFACTS["hf_available"] = False
    _ARTIFACTS["hf_download_error"] = None

    local_resolved = models_path.resolve() if models_path.exists() else models_path

    def _label_source(local: bool) -> str:
        return "local" if local else "huggingface_hub"

    # intent label encoder
    local_path = _resolve_artifact_local("intent_label_encoder.joblib", models_path)
    intent_label_path = local_path or _hf_download("intent_label_encoder.joblib")
    if intent_label_path is not None:
        try:
            import joblib  # noqa: WPS433
            _ARTIFACTS["intent_label_encoder"] = joblib.load(intent_label_path)
            _ARTIFACTS["torch_intent_label_encoder_source"] = _label_source(local_path is not None)
            logger.info(
                "Загружен intent_label_encoder.joblib (%s: %s)",
                _ARTIFACTS["torch_intent_label_encoder_source"], intent_label_path,
            )
            if local_path is None:
                _ARTIFACTS["hf_available"] = True
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось загрузить intent_label_encoder.joblib: %s", exc)

    # старый sklearn-style intent_model (оставлен для обратной совместимости)
    intent_model_path = _resolve_artifact_local("intent_model.joblib", models_path)
    if intent_model_path is not None:
        try:
            import joblib  # noqa: WPS433
            _ARTIFACTS["intent_model"] = joblib.load(intent_model_path)
            logger.info("Загружен intent_model.joblib: %s", intent_model_path)
        except Exception as exc:  # pragma: no cover - зависит от артефактов
            logger.warning("Не удалось загрузить intent_model.joblib: %s", exc)

    # multitask config
    local_path = _resolve_artifact_local("multitask_config.json", models_path)
    config_path = local_path or _hf_download("multitask_config.json")
    if config_path is not None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _ARTIFACTS["torch_intent_config"] = json.load(f)
            _ARTIFACTS["torch_intent_config_source"] = _label_source(local_path is not None)
            logger.info(
                "Загружен multitask_config.json (%s): model_name=%s, num_intents=%s",
                _ARTIFACTS["torch_intent_config_source"],
                _ARTIFACTS["torch_intent_config"].get("model_name"),
                _ARTIFACTS["torch_intent_config"].get("num_intents"),
            )
            if local_path is None:
                _ARTIFACTS["hf_available"] = True
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось прочитать multitask_config.json: %s", exc)

    # single-task RuBERT state_dict
    intent_state_file = os.environ.get("INTENT_MODEL_FILE", "single_task_intent_model.pt")
    local_path = _resolve_artifact_local(intent_state_file, models_path)
    intent_state_path = local_path or _hf_download(intent_state_file)
    if intent_state_path is not None:
        _ARTIFACTS["torch_intent_state_path"] = str(intent_state_path.resolve())
        _ARTIFACTS["torch_intent_state_source"] = _label_source(local_path is not None)
        logger.info(
            "Обнаружен single-task RuBERT state_dict (%s): %s (lazy-load при первом predict_intent)",
            _ARTIFACTS["torch_intent_state_source"], intent_state_path,
        )
        if local_path is None:
            _ARTIFACTS["hf_available"] = True

    # topic centroids (только локально; не входят в HF репо)
    centroids_path = _resolve_artifact_local("topic_centroids.npy", models_path)
    if centroids_path is not None:
        try:
            _ARTIFACTS["topic_centroids"] = np.load(centroids_path)
            logger.info("Загружен topic_centroids.npy: shape=%s", _ARTIFACTS["topic_centroids"].shape)
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось загрузить topic_centroids.npy: %s", exc)

    # topic metadata
    meta_path = _resolve_artifact_local("topic_metadata.parquet", models_path)
    if meta_path is not None:
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

    intent_state_source = _ARTIFACTS.get("torch_intent_state_source") or (
        "missing" if not has_torch_state else "unknown"
    )

    summarization_enabled = _env_flag("ENABLE_SUMMARIZATION", True)
    summarizer_loaded = _ARTIFACTS.get("summarizer_model") is not None
    summary_source = _ARTIFACTS.get("summarizer_source")
    if not summarization_enabled:
        summary_mode = "disabled"
    elif _ARTIFACTS.get("summarizer_load_error"):
        summary_mode = "error"
    elif summarizer_loaded:
        summary_mode = "transformers_seq2seq"
    else:
        summary_mode = "pending_lazy_load"

    return {
        "models_dir": _ARTIFACTS.get("models_dir"),
        "hf_repo_id": _ARTIFACTS.get("hf_repo_id"),
        "hf_available": _ARTIFACTS.get("hf_available", False),
        "hf_download_error": _ARTIFACTS.get("hf_download_error"),
        "torch_intent_enabled": enabled,
        "torch_intent_state_found": has_torch_state,
        "torch_intent_state_path": _ARTIFACTS.get("torch_intent_state_path"),
        "torch_intent_state_source": intent_state_source,
        "intent_label_encoder_found": has_encoder,
        "intent_label_encoder_source": _ARTIFACTS.get("torch_intent_label_encoder_source"),
        "multitask_config_found": has_config,
        "multitask_config_source": _ARTIFACTS.get("torch_intent_config_source"),
        "torch_intent_runtime_loaded": runtime_loaded,
        "torch_intent_load_error": _ARTIFACTS.get("torch_intent_load_error"),
        "intent_mode_planned": intent_mode,
        "topic_centroids_found": _ARTIFACTS.get("topic_centroids") is not None,
        "topic_metadata_found": _ARTIFACTS.get("topic_metadata") is not None,
        "summarization_enabled": summarization_enabled,
        "summarizer_loaded": summarizer_loaded,
        "summarizer_source": summary_source,
        "summarizer_path_or_name": _ARTIFACTS.get("summarizer_path_or_name"),
        "summarizer_device": _ARTIFACTS.get("summarizer_device"),
        "summarizer_load_error": _ARTIFACTS.get("summarizer_load_error"),
        "summary_mode": summary_mode,
    }


def _ensure_loaded() -> None:
    if not _ARTIFACTS["loaded"]:
        load_artifacts(os.environ.get("MODELS_DIR", "models"))


# ---------------------------------------------------------------------------
# Single-task RuBERT runtime
# ---------------------------------------------------------------------------

_DEFAULT_TORCH_MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
_DEFAULT_TORCH_MAX_LEN = 128
_DEFAULT_TORCH_DROPOUT = 0.1


class SingleTaskIntentModelRuntime:
    """Runtime-обёртка `SingleTaskIntentModel`.

    Архитектура повторяет `SingleTaskIntentModel`:
        encoder = AutoModel.from_pretrained(base_model_name)
        shared/proj = Linear(h, h) -> GELU -> LayerNorm -> Dropout
        intent_head = Linear(h, num_intents)
    Pooling — mean по attention mask.

    Имя поля проекции в state_dict — `proj.*`. На случай чужих чекпойнтов
    поддерживается `load_state_dict(strict=False)` с предупреждением.
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
            state = state["state_dict"]

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
    """Ленивая инициализация single-task RuBERT runtime."""
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

    Порядок: single-task RuBERT runtime → старый sklearn joblib intent_model
    → rule-based fallback.
    """
    _ensure_loaded()
    if not text or not text.strip():
        return {"label": "other", "confidence": 0.0, "mode": "empty_input"}

    encoder = _ARTIFACTS.get("intent_label_encoder")

    # 1) Single-task RuBERT runtime
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
    """Предсказывает тематический кластер."""
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
                norms = np.linalg.norm(cents, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                cents_n = cents / norms
                sims = cents_n @ emb
                idx = int(np.argmax(sims))
                confidence = float((sims[idx] + 1.0) / 2.0)
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
# Summarization (HuggingFace Transformers, ленивая загрузка)
# ---------------------------------------------------------------------------

_DEFAULT_SUMMARIZATION_MODEL = "IlyaGusev/rut5_base_sum_gazeta"


def _summarizer_local_dirs() -> List[Path]:
    """Локальные кандидаты для папки суммаризатора."""
    candidates: List[Path] = []
    explicit = os.environ.get("SUMMARIZATION_LOCAL_DIR")
    if explicit:
        candidates.append(Path(explicit))
    models_dir = Path(_ARTIFACTS.get("models_dir") or os.environ.get("MODELS_DIR", "models"))
    candidates.append(models_dir / "summarizer")
    candidates.append(models_dir / "summarization")
    return candidates


def _resolve_summarizer_source() -> Dict[str, Any]:
    """Определяет, откуда грузить summarizer: локально или с HF Hub."""
    for d in _summarizer_local_dirs():
        try:
            if d.exists() and (d / "config.json").exists():
                return {"path_or_name": str(d.resolve()), "source": "local"}
        except OSError:
            continue
    model_name = os.environ.get("SUMMARIZATION_MODEL_NAME", _DEFAULT_SUMMARIZATION_MODEL)
    return {"path_or_name": model_name, "source": "huggingface_hub"}


def _build_summarizer() -> bool:
    """Ленивая инициализация summarization-модели. True при успехе."""
    if _ARTIFACTS.get("summarizer_model") is not None:
        return True
    if _ARTIFACTS.get("summarizer_attempted") and _ARTIFACTS.get("summarizer_load_error"):
        return False
    _ARTIFACTS["summarizer_attempted"] = True

    if not _env_flag("ENABLE_SUMMARIZATION", True):
        _ARTIFACTS["summarizer_load_error"] = "ENABLE_SUMMARIZATION=false"
        return False

    resolved = _resolve_summarizer_source()
    path_or_name = resolved["path_or_name"]
    source = resolved["source"]
    _ARTIFACTS["summarizer_path_or_name"] = path_or_name
    _ARTIFACTS["summarizer_source"] = source

    try:
        import torch  # noqa: WPS433
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: WPS433
    except Exception as exc:
        _ARTIFACTS["summarizer_load_error"] = f"transformers/torch недоступны: {exc}"
        logger.warning("Summarizer: %s", _ARTIFACTS["summarizer_load_error"])
        return False

    try:
        logger.info("Summarizer: загрузка tokenizer/model из %s (source=%s)", path_or_name, source)
        tokenizer = AutoTokenizer.from_pretrained(path_or_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(path_or_name)
    except Exception as exc:
        _ARTIFACTS["summarizer_load_error"] = f"load failed: {exc}"
        logger.warning("Summarizer load_error: %s", exc)
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda":
            try:
                model = model.to(device, dtype=torch.float16)
            except Exception:
                model = model.to(device)
        else:
            model = model.to(device)
        model.eval()
    except Exception as exc:
        _ARTIFACTS["summarizer_load_error"] = f"device move failed: {exc}"
        logger.warning("Summarizer device move: %s", exc)
        return False

    _ARTIFACTS["summarizer_tokenizer"] = tokenizer
    _ARTIFACTS["summarizer_model"] = model
    _ARTIFACTS["summarizer_device"] = device
    _ARTIFACTS["summarizer_load_error"] = None
    logger.info("Summarizer готов: %s on %s", path_or_name, device)
    return True


def summarize(text: str):
    """Возвращает кортеж (summary, status)."""
    if not text or not text.strip():
        return None, "Пустой ввод — суммаризация не выполнена"

    if not _env_flag("ENABLE_SUMMARIZATION", True):
        return None, "Суммаризация отключена (ENABLE_SUMMARIZATION=false)"

    if not _build_summarizer():
        err = _ARTIFACTS.get("summarizer_load_error") or "неизвестная ошибка"
        return None, f"Суммаризация недоступна: {err}"

    tokenizer = _ARTIFACTS["summarizer_tokenizer"]
    model = _ARTIFACTS["summarizer_model"]
    device = _ARTIFACTS["summarizer_device"]

    try:
        import torch  # noqa: WPS433
    except Exception as exc:  # pragma: no cover
        return None, f"Суммаризация недоступна: torch отсутствует ({exc})"

    max_input = int(os.environ.get("SUMMARIZATION_MAX_INPUT_TOKENS", "1024"))
    max_new = int(os.environ.get("SUMMARIZATION_MAX_NEW_TOKENS", "96"))
    num_beams = int(os.environ.get("SUMMARIZATION_NUM_BEAMS", "4"))

    try:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_input,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        summary = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    except Exception as exc:
        logger.exception("Суммаризация упала: %s", exc)
        return None, f"Ошибка суммаризации: {exc}"

    source = _ARTIFACTS.get("summarizer_source")
    path_or_name = _ARTIFACTS.get("summarizer_path_or_name")
    status = f"Суммаризация выполнена ({source}): {path_or_name}"
    return summary, status


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
    import json
    load_artifacts(os.environ.get("MODELS_DIR", "models"))
    sample = "Здравствуйте, я хочу забронировать билет на концерт в Москве"
    print(json.dumps(analyze(sample), ensure_ascii=False, indent=2))
