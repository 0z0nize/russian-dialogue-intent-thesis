# Ноутбуки

Эти Jupyter-ноутбуки реализуют подготовку данных, обучение, оценку и анализ результатов для выпускной квалификационной работы (ВКР) по распознаванию намерений в русскоязычных диалогах.

Рекомендуемый порядок просмотра соответствует числовым префиксам в именах файлов (`00`, `01`, …, `11`): ноутбуки 00–09 образуют основной экспериментальный конвейер, ноутбук 10 — вспомогательный инструмент ручной валидации интентов, ноутбук 11 — нейросетевое многозадачное моделирование и финальная модель для веб-приложения.

> Если GitHub не отображает `.ipynb` (ошибка рендеринга или «Sorry, something went wrong»), откройте ноутбук через [nbviewer](https://nbviewer.org/) по соответствующей ссылке из таблицы ниже.

| № | Файл | Назначение | Просмотр |
| --- | --- | --- | --- |
| 00 | [`00_create_drive_folders.ipynb`](00_create_drive_folders.ipynb) | Создание структуры папок проекта в Google Drive (служебно) | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/00_create_drive_folders.ipynb) |
| 01 | [`01_data_loading_and_eda.ipynb`](01_data_loading_and_eda.ipynb) | Первичная загрузка данных и EDA | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/01_data_loading_and_eda.ipynb) |
| 02 | [`02_annotation_prep.ipynb`](02_annotation_prep.ipynb) | Подготовка материалов для разметки | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/02_annotation_prep.ipynb) |
| 03 | [`03_annotation_sanity_check.ipynb`](03_annotation_sanity_check.ipynb) | Контроль качества разметки | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/03_annotation_sanity_check.ipynb) |
| 04 | [`04_baseline_modeling.ipynb`](04_baseline_modeling.ipynb) | Baselines для распознавания намерений | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/04_baseline_modeling.ipynb) |
| 05 | [`05_hierarchical_intent_modeling.ipynb`](05_hierarchical_intent_modeling.ipynb) | Иерархическое моделирование интентов | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/05_hierarchical_intent_modeling.ipynb) |
| 06 | [`06_english_translation_intent_modeling.ipynb`](06_english_translation_intent_modeling.ipynb) | Эксперимент с переводом на английский | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/06_english_translation_intent_modeling.ipynb) |
| 07 | [`07_dialogsum_ru_eda.ipynb`](07_dialogsum_ru_eda.ipynb) | EDA на корпусе DialogSum-RU (глава 4) | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/07_dialogsum_ru_eda.ipynb) |
| 08 | [`08_topic_modeling_dialogsum_ru.ipynb`](08_topic_modeling_dialogsum_ru.ipynb) | Тематическое моделирование: embeddings + UMAP + HDBSCAN, 208 кластеров, silhouette ≈ 0,7623 (глава 5) | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/08_topic_modeling_dialogsum_ru.ipynb) |
| 09 | [`09_intent_modeling_dialogsum_ru.ipynb`](09_intent_modeling_dialogsum_ru.ipynb) | Распознавание намерений: слабая разметка, baselines, RuBERT-tiny2 (главы 6–7) | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/09_intent_modeling_dialogsum_ru.ipynb) |
| 10 | [`10_intent_manual_validation_dialogsum_ru.ipynb`](10_intent_manual_validation_dialogsum_ru.ipynb) | Инструмент ручной валидации интентов | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/10_intent_manual_validation_dialogsum_ru.ipynb) |
| 11 | [`11_neural_multitask_intent_topic_dialogsum_ru.ipynb`](11_neural_multitask_intent_topic_dialogsum_ru.ipynb) | Нейросетевое моделирование на RuBERT-base: однозадачная vs многозадачная архитектура; источник финальной модели для веб-приложения | [nbviewer](https://nbviewer.org/github/0z0nize/russian-dialogue-intent-thesis/blob/main/notebooks/11_neural_multitask_intent_topic_dialogsum_ru.ipynb) |
