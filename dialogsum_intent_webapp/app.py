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
    default_summary_model = os.environ.get(
        "SUMMARIZATION_MODEL_NAME", "IlyaGusev/rut5_base_sum_gazeta"
    )
    summary_target = (
        status.get("summarizer_path_or_name")
        or (default_summary_model if summary_mode != "disabled" else "—")
    )
    summary_source = status.get("summarizer_source")
    if summary_mode == "pending_lazy_load":
        summary_source_label = (
            "будет загружена при первом запросе на суммаризацию"
        )
        mode_label = "ожидает первого запроса (ленивая загрузка)"
    elif summary_mode == "transformers_seq2seq":
        src_map = {"local": "локальная папка", "huggingface_hub": "Hugging Face Hub"}
        summary_source_label = src_map.get(summary_source or "", summary_source or "—")
        mode_label = "готова (transformers seq2seq)"
    elif summary_mode == "disabled":
        summary_source_label = "—"
        mode_label = "отключена (ENABLE_SUMMARIZATION=false)"
    else:
        summary_source_label = summary_source or "—"
        mode_label = summary_mode
    summary_err = status.get("summarizer_load_error")
    summary_err_md = (
        f"<br><span class='small-note'>⚠ Суммаризация: {summary_err}</span>"
        if summary_err else ""
    )
    summary_line = (
        f"<div class='small-note'>{summary_icon} <b>Суммаризация:</b> "
        f"{mode_label} · модель: <code>{summary_target}</code> "
        f"(источник: {summary_source_label}){summary_err_md}</div>"
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

# Инлайновый ITMO-логотип: компактный SVG, чтобы UI не зависел от внешних
# картинок (не падал без сети и не упирался в политики CSP площадок).
ITMO_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" width="64" height="64" aria-label="ITMO" role="img">
  <rect x="2" y="2" width="92" height="92" rx="14" fill="#0f4ea8"/>
  <text x="48" y="44" text-anchor="middle"
        font-family="'Helvetica Neue', Arial, sans-serif"
        font-weight="700" font-size="26" fill="#ffffff" letter-spacing="2">ITMO</text>
  <text x="48" y="66" text-anchor="middle"
        font-family="'Helvetica Neue', Arial, sans-serif"
        font-weight="500" font-size="11" fill="#cfe1ff" letter-spacing="1.5">UNIVERSITY</text>
  <rect x="14" y="74" width="68" height="6" rx="3" fill="#f5a623"/>
</svg>
"""

HEADER_HTML = f"""
<div class="itmo-header">
  <div class="itmo-logo">{ITMO_LOGO_SVG}</div>
  <div class="itmo-title">
    <h1>Семантический анализ русскоязычных диалогов для задачи распознавания намерений с улучшением на базе предобученных моделей</h1>
    <div class="itmo-meta">
      <b>Шкаровский Владислав Семёнович</b> · НИУ ИТМО, магистратура, образовательная программа «Аналитика данных».
    </div>
    <div class="itmo-meta">
      <a href="https://github.com/0z0nize/russian-dialogue-intent-thesis/blob/main/README.md" target="_blank" rel="noopener">📖 README на GitHub</a>
    </div>
  </div>
</div>
"""

INTRO_MD = """
Демо к магистерской ВКР по корпусу **DialogSum-RU**: на вход — текст или
голосовая запись на русском, на выход — намерение (intent), тематический
кластер и сводка по реплике.

Если артефакты моделей не подключены, используются rule-based fallback
из ВКР, чтобы интерфейс работал даже на пустом VPS.
"""

# Темы и примеры для тестирования интерфейса. Показываются под кнопками
# «Анализировать» / «Распознать», чтобы пользователь видел подсказку, но
# она не загромождала верх страницы.
TEST_THEMES_HTML = """
<details class="itmo-hints" open>
  <summary>💡 Темы для тестирования интерфейса (нажмите, чтобы свернуть)</summary>
  <div class="itmo-hints-body">
    <div>Попробуйте реплики на следующие темы:</div>
    <ul class="itmo-themes">
      <li>собеседование / работа</li>
      <li>путешествия / билеты</li>
      <li>покупка / заказ</li>
      <li>ремонт / обслуживание</li>
      <li>жалоба / проблема</li>
      <li>образование</li>
      <li>дом / бытовые вопросы</li>
      <li>поиск книг</li>
      <li>музыкальные события</li>
      <li>развлечения</li>
    </ul>
    <div class="itmo-hints-examples">
      <b>Готовые примеры:</b>
      <ul>
        <li>«Здравствуйте, я хочу забронировать билет на концерт в Москве.»</li>
        <li>«Подскажите, как записаться на собеседование по вакансии аналитика.»</li>
        <li>«У меня жалоба: посылка пришла повреждённой, верните деньги.»</li>
        <li>«Не могу найти книгу автора Достоевского, помогите подобрать издание.»</li>
        <li>«Нужно вызвать мастера — стиральная машина не работает.»</li>
      </ul>
    </div>
  </div>
</details>
"""

CSS = """
.gradio-container {max-width: 1100px !important;}
#json-out textarea, #json-out pre {font-size: 0.85rem;}
.small-note {color: #6b6b6b; font-size: 0.85rem;}
.itmo-header {display: flex; align-items: center; gap: 18px; margin: 4px 0 8px 0;}
.itmo-header .itmo-logo {flex: 0 0 auto;}
.itmo-header .itmo-title h1 {margin: 0 0 6px 0; font-size: 1.25rem; line-height: 1.3;}
.itmo-header .itmo-meta {font-size: 0.9rem; color: #444;}
.itmo-hints {background: #f4f7fb; border: 1px solid #d6e1ef; border-radius: 8px;
             padding: 8px 12px; margin: 6px 0 4px 0; font-size: 0.9rem;}
.itmo-hints summary {cursor: pointer; font-weight: 600; color: #0f4ea8;}
.itmo-hints .itmo-themes {columns: 2; -webkit-columns: 2; -moz-columns: 2;
                          margin: 6px 0 6px 18px; padding: 0;}
.itmo-hints .itmo-themes li {margin: 1px 0;}
.itmo-hints-examples {margin-top: 6px;}
.itmo-hints-examples ul {margin: 4px 0 0 18px; padding: 0;}
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
        gr.HTML(HEADER_HTML)
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
                        gr.HTML(TEST_THEMES_HTML)
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
                        gr.HTML(
                            "<details class='itmo-hints'>"
                            "<summary>💡 Темы для тестирования</summary>"
                            "<div class='itmo-hints-body'>Произнесите фразу на одну из тем: "
                            "собеседование / работа, путешествия / билеты, покупка / заказ, "
                            "ремонт / обслуживание, жалоба / проблема, образование, "
                            "дом / бытовые вопросы, поиск книг, музыкальные события, развлечения."
                            "</div></details>"
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
