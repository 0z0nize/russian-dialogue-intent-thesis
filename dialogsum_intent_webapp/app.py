"""
Gradio Blocks UI для демо ВКР по русскоязычному распознаванию намерений
и тематическому моделированию DialogSum-RU.

Запуск:
    python app.py
Откроется на http://0.0.0.0:7860
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Tuple

import gradio as gr

import model_pipeline as mp

logger = logging.getLogger(__name__)

# Прогрузим артефакты сразу при импорте (whisper всё равно ленив)
mp.load_artifacts(os.environ.get("MODELS_DIR", "models"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_top_words(words) -> str:
    if not words:
        return ""
    return ", ".join(str(w) for w in words)


def _result_tuple(result: Dict[str, Any]) -> Tuple:
    """Раскладывает dict от analyze() в кортеж для Gradio outputs."""
    return (
        result.get("intent_label") or "",
        result.get("intent_confidence") if result.get("intent_confidence") is not None else 0.0,
        result.get("topic_cluster_id") if result.get("topic_cluster_id") is not None else -1,
        result.get("topic_cluster_name") or "",
        result.get("topic_cluster_description") or "",
        _format_top_words(result.get("topic_top_words")),
        result.get("summary") if result.get("summary") is not None else (result.get("summary_status") or ""),
        result,
    )


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
        empty = mp.analyze("")
        return ("",) + _result_tuple(empty)
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
# Семантический анализ русскоязычных диалогов

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


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="DialogSum-RU · intent demo", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO_MD)

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
                        text_btn = gr.Button("Анализировать", variant="primary")
                        gr.Markdown(
                            "<span class='small-note'>Если артефакты моделей отсутствуют — сработает rule-based fallback.</span>"
                        )
                    with gr.Column(scale=4):
                        t_intent = gr.Textbox(label="Intent (намерение)")
                        t_intent_conf = gr.Number(label="Intent confidence", precision=3)
                        with gr.Row():
                            t_topic_id = gr.Number(label="Topic cluster id", precision=0)
                            t_topic_name = gr.Textbox(label="Topic name")
                        t_topic_desc = gr.Textbox(label="Topic description", lines=2)
                        t_topic_words = gr.Textbox(label="Top words")
                        t_summary = gr.Textbox(label="Summary / статус", lines=2)
                        t_json = gr.JSON(label="Полный JSON-ответ", elem_id="json-out")

                text_btn.click(
                    analyze_text,
                    inputs=[text_input],
                    outputs=[
                        t_intent,
                        t_intent_conf,
                        t_topic_id,
                        t_topic_name,
                        t_topic_desc,
                        t_topic_words,
                        t_summary,
                        t_json,
                    ],
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
                        audio_btn = gr.Button("Распознать и анализировать", variant="primary")
                        gr.Markdown(
                            "<span class='small-note'>STT: faster-whisper. По умолчанию модель "
                            f"<code>{os.environ.get('WHISPER_MODEL_SIZE', 'medium')}</code>. "
                            "Микрофон в браузере обычно требует HTTPS — иначе используйте upload.</span>"
                        )
                    with gr.Column(scale=4):
                        a_text = gr.Textbox(label="Распознанный текст", lines=4)
                        a_intent = gr.Textbox(label="Intent (намерение)")
                        a_intent_conf = gr.Number(label="Intent confidence", precision=3)
                        with gr.Row():
                            a_topic_id = gr.Number(label="Topic cluster id", precision=0)
                            a_topic_name = gr.Textbox(label="Topic name")
                        a_topic_desc = gr.Textbox(label="Topic description", lines=2)
                        a_topic_words = gr.Textbox(label="Top words")
                        a_summary = gr.Textbox(label="Summary / статус", lines=2)
                        a_json = gr.JSON(label="Полный JSON-ответ", elem_id="json-out")

                audio_btn.click(
                    analyze_audio,
                    inputs=[audio_input],
                    outputs=[
                        a_text,
                        a_intent,
                        a_intent_conf,
                        a_topic_id,
                        a_topic_name,
                        a_topic_desc,
                        a_topic_words,
                        a_summary,
                        a_json,
                    ],
                )

        gr.Markdown(
            "<span class='small-note'>Проект: ВКР «Семантический анализ русскоязычных диалогов "
            "для распознавания намерений», НИУ ИТМО. Демо запускается через Gradio Blocks.</span>"
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
