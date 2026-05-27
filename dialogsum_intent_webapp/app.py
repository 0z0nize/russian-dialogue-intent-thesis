"""
Gradio Blocks UI для демо ВКР по русскоязычному распознаванию намерений
и тематическому моделированию DialogSum-RU.

Запуск:
    python app.py
Откроется на http://0.0.0.0:7860
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

import gradio as gr

import model_pipeline as mp

logger = logging.getLogger(__name__)

# Прогрузим артефакты сразу при импорте (whisper и RuBERT всё равно ленивы)
mp.load_artifacts(os.environ.get("MODELS_DIR", "models"))


def _status_banner_md() -> str:
    status = mp.get_artifact_status()
    mode = status.get("intent_mode_planned", "rule_based_fallback")
    icon = "✅" if mode == "single_task_rubert_model" else (
        "🟡" if mode == "sklearn_intent_model" else "⚠️"
    )
    if mode == "single_task_rubert_model":
        intent_text = (
            "Single-task RuBERT (обученные артефакты проекта). "
            "Test metrics: accuracy 0.9126, macro-F1 0.7770."
        )
    elif mode == "sklearn_intent_model":
        intent_text = "sklearn-pipeline intent_model.joblib (legacy)."
    else:
        intent_text = (
            "Артефакты не подключены — работает rule-based fallback. "
            "См. README → «Подключение артефактов из Hugging Face Hub»."
        )

    intent_src = status.get("torch_intent_state_source") or "missing"
    src_label = {
        "local": "локальная папка models/",
        "huggingface_hub": "Hugging Face Hub",
        "missing": "артефакт не найден",
    }.get(intent_src, intent_src)
    hf_repo = status.get("hf_repo_id") or "—"
    src_note = (
        f" · <b>Источник intent:</b> <code>{intent_src}</code> ({src_label})"
        f" · <b>HF репозиторий:</b> <code>{hf_repo}</code>"
    )

    err = status.get("torch_intent_load_error")
    err_md = f"<br><span class='small-note'>⚠ Загрузка torch модели: {err}</span>" if err else ""
    hf_err = status.get("hf_download_error")
    hf_err_md = (
        f"<br><span class='small-note'>⚠ Hugging Face Hub: {hf_err}</span>"
        if hf_err else ""
    )

    summary_mode = status.get("summary_mode", "pending_lazy_load")
    summary_icon = {
        "transformers_seq2seq": "✅",
        "pending_lazy_load": "🕓",
        "disabled": "⛔",
        "error": "⚠️",
    }.get(summary_mode, "ℹ️")
    summary_target = status.get("summarizer_path_or_name") or "не выбран"
    summary_source = status.get("summarizer_source") or "будет определён при первом вызове"
    summary_err = status.get("summarizer_load_error")
    summary_err_md = (
        f"<br><span class='small-note'>⚠ Суммаризация: {summary_err}</span>"
        if summary_err else ""
    )
    summary_line = (
        f"<div class='small-note'>{summary_icon} <b>Summary mode:</b> "
        f"<code>{summary_mode}</code> · модель: <code>{summary_target}</code> "
        f"(источник: <code>{summary_source}</code>){summary_err_md}</div>"
    )

    return (
        f"<div class='small-note'>{icon} <b>Intent mode:</b> "
        f"<code>{mode}</code> — {intent_text}{src_note}{err_md}{hf_err_md}</div>"
        + summary_line
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_top_words(words) -> str:
    if not words:
        return ""
    return ", ".join(str(w) for w in words)


def _format_summary(result: Dict[str, Any]) -> str:
    summary = result.get("summary")
    status = result.get("summary_status") or ""
    if summary:
        return f"{summary}\n\n[{status}]" if status else summary
    return status


def _result_tuple(result: Dict[str, Any]) -> Tuple:
    """Раскладывает dict от analyze() в кортеж для Gradio outputs."""
    return (
        result.get("intent_label") or "",
        result.get("intent_confidence") if result.get("intent_confidence") is not None else 0.0,
        result.get("intent_mode") or "",
        result.get("topic_cluster_id") if result.get("topic_cluster_id") is not None else -1,
        result.get("topic_cluster_name") or "",
        result.get("topic_cluster_description") or "",
        _format_top_words(result.get("topic_top_words")),
        _format_summary(result),
        result,
    )


def _empty_result_tuple() -> Tuple:
    """Кортеж для очистки полей результата."""
    return ("", 0.0, "", -1, "", "", "", "", None)


def analyze_text(text: str):
    try:
        result = mp.analyze(text or "")
    except Exception as exc:
        logger.exception("analyze ошибка: %s", exc)
        result = {
            "input_text": text or "",
            "intent_label": "other",
            "intent_confidence": 0.0,
            "intent_mode": f"error: {exc}",
            "topic_cluster_id": -1,
            "topic_cluster_name": "не определена",
            "topic_cluster_description": str(exc),
            "topic_top_words": [],
            "topic_confidence": 0.0,
            "topic_mode": f"error: {exc}",
            "summary": None,
            "summary_status": "Ошибка при анализе",
        }
    return _result_tuple(result)


def analyze_audio(audio_path: str):
    if not audio_path:
        return ("",) + _empty_result_tuple()
    try:
        text = mp.transcribe_audio(audio_path)
    except Exception as exc:
        logger.exception("transcribe ошибка: %s", exc)
        text = f"[Ошибка распознавания: {exc}]"
    result_tuple = analyze_text(text)
    return (text,) + result_tuple


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INTRO_MD = """
# Семантический анализ русскоязычных диалогов для задачи распознавания намерений с улучшением на базе предобученных моделей

**Шкаровский Владислав Семёнович**, НИУ ИТМО, магистратура, образовательная программа «Аналитика данных».

[📖 README на GitHub](https://github.com/0z0nize/russian-dialogue-intent-thesis/blob/main/README.md)

Демо к магистерской ВКР по корпусу **DialogSum-RU**: на вход — текст или
голосовая запись на русском, на выход — намерение (intent), тематический
кластер и сводка по реплике.

Если артефакты моделей не подключены, используются rule-based fallback
из ВКР, чтобы интерфейс работал даже на пустом VPS.
"""

CSS = """
.gradio-container {max-width: 1100px !important;}
#json-out textarea, #json-out pre {font-size: 0.85rem;}
.small-note {color: #6b6b6b; font-size: 0.85rem;}
"""


def _gradio_major_version() -> int:
    try:
        return int(gr.__version__.split(".", 1)[0])
    except (AttributeError, ValueError):
        return 0


def build_demo() -> gr.Blocks:
    # В Gradio 6 параметры theme/css в конструкторе Blocks вызывают
    # UserWarning и должны передаваться в demo.launch(). В Gradio 5
    # launch() их не принимает — там оставляем старый путь.
    blocks_kwargs: Dict[str, Any] = {"title": "DialogSum-RU · intent demo"}
    if _gradio_major_version() < 6:
        blocks_kwargs["css"] = CSS
        blocks_kwargs["theme"] = gr.themes.Soft()
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(INTRO_MD)
        gr.Markdown(_status_banner_md())

        with gr.Tabs():
            # ---------------- Tab 1: Текст ----------------
            with gr.Tab("Текст"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Введите реплику или короткий диалог",
                            placeholder="Например: Здравствуйте, я хочу забронировать билет на концерт в Москве.",
                            lines=6,
                        )
                        with gr.Row():
                            text_btn = gr.Button("Анализировать", variant="primary")
                            text_clear_btn = gr.Button("Очистить", variant="secondary")
                        gr.Markdown(
                            "<span class='small-note'>Поле <b>Intent mode</b> ниже показывает, "
                            "какой модуль обработал реплику: <code>single_task_rubert_model</code> "
                            "(обученные артефакты проекта) или <code>rule_based_fallback</code>.</span>"
                        )
                    with gr.Column(scale=4):
                        t_intent = gr.Textbox(label="Intent (намерение)")
                        t_intent_conf = gr.Number(label="Intent confidence", precision=3)
                        t_intent_mode = gr.Textbox(label="Intent mode (источник предсказания)")
                        with gr.Row():
                            t_topic_id = gr.Number(label="Topic cluster id", precision=0)
                            t_topic_name = gr.Textbox(label="Topic name")
                        t_topic_desc = gr.Textbox(label="Topic description", lines=2)
                        t_topic_words = gr.Textbox(label="Top words")
                        t_summary = gr.Textbox(label="Summary / статус", lines=2)
                        t_json = gr.JSON(label="Полный JSON-ответ", elem_id="json-out")

                text_outputs = [
                    t_intent,
                    t_intent_conf,
                    t_intent_mode,
                    t_topic_id,
                    t_topic_name,
                    t_topic_desc,
                    t_topic_words,
                    t_summary,
                    t_json,
                ]

                text_btn.click(
                    analyze_text,
                    inputs=[text_input],
                    outputs=text_outputs,
                )

                text_clear_btn.click(
                    lambda: ("",) + _empty_result_tuple(),
                    inputs=None,
                    outputs=[text_input] + text_outputs,
                )

            # ---------------- Tab 2: Голос ----------------
            with gr.Tab("Голос"):
                with gr.Row():
                    with gr.Column(scale=3):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Запишите или загрузите аудио (русский)",
                        )
                        with gr.Row():
                            audio_btn = gr.Button("Распознать и анализировать", variant="primary")
                            audio_clear_btn = gr.Button("Очистить голосовой ввод", variant="secondary")
                        gr.Markdown(
                            "<span class='small-note'>STT: faster-whisper. По умолчанию модель "
                            f"<code>{os.environ.get('WHISPER_MODEL_SIZE', 'medium')}</code>. "
                            "Микрофон в браузере обычно требует HTTPS — иначе используйте upload.</span>"
                        )
                    with gr.Column(scale=4):
                        a_text = gr.Textbox(label="Распознанный текст", lines=4)
                        a_intent = gr.Textbox(label="Intent (намерение)")
                        a_intent_conf = gr.Number(label="Intent confidence", precision=3)
                        a_intent_mode = gr.Textbox(label="Intent mode (источник предсказания)")
                        with gr.Row():
                            a_topic_id = gr.Number(label="Topic cluster id", precision=0)
                            a_topic_name = gr.Textbox(label="Topic name")
                        a_topic_desc = gr.Textbox(label="Topic description", lines=2)
                        a_topic_words = gr.Textbox(label="Top words")
                        a_summary = gr.Textbox(label="Summary / статус", lines=2)
                        a_json = gr.JSON(label="Полный JSON-ответ", elem_id="json-out")

                audio_outputs = [
                    a_text,
                    a_intent,
                    a_intent_conf,
                    a_intent_mode,
                    a_topic_id,
                    a_topic_name,
                    a_topic_desc,
                    a_topic_words,
                    a_summary,
                    a_json,
                ]

                audio_btn.click(
                    analyze_audio,
                    inputs=[audio_input],
                    outputs=audio_outputs,
                )

                audio_clear_btn.click(
                    lambda: (None, "") + _empty_result_tuple(),
                    inputs=None,
                    outputs=[audio_input] + audio_outputs,
                )

        gr.Markdown(
            "<span class='small-note'>Проект: ВКР «Семантический анализ русскоязычных диалогов "
            "для распознавания намерений», НИУ ИТМО. Демо запускается через Gradio Blocks.</span>"
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue()
    launch_kwargs: Dict[str, Any] = dict(
        server_name="0.0.0.0", server_port=7860, share=False
    )
    if _gradio_major_version() >= 6:
        launch_kwargs["theme"] = gr.themes.Soft()
        launch_kwargs["css"] = CSS
    demo.launch(**launch_kwargs)
