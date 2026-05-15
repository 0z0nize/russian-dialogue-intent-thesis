# Russian Dialogue Intent Thesis

## Описание проекта

Проект подготовлен для магистерской диссертации на тему: «Семантический анализ русскоязычных диалогов для задачи распознавания намерений с улучшением на базе предобученных моделей».

Цель проекта — подготовить и исследовать подходы к распознаванию намерений в русскоязычных диалогах с использованием предобученных языковых моделей и последующим сравнением качества различных решений.

Основной датасет: `dialogsum-ru`.

Предполагаемая постановка задачи: классификация интента всего диалога.

Возможный baseline: перевод русскоязычных диалогов на английский язык и применение англоязычной модели для классификации намерений.

## Структура репозитория

- `configs/` — конфигурации данных, моделей и экспериментов.
- `data/` — локальные служебные директории для данных; сами данные не хранятся в Git.
- `data/raw/` — исходные данные.
- `data/interim/` — промежуточные данные.
- `data/processed/` — обработанные данные.
- `data/annotation/` — материалы разметки.
- `notebooks/` — исследовательские notebooks.
- `src/` — исходный код проекта.
- `src/data/` — подготовка и загрузка данных.
- `src/features/` — извлечение и подготовка признаков.
- `src/models/` — обучение и применение моделей.
- `src/evaluation/` — оценка качества моделей.
- `src/utils/` — вспомогательные функции.
- `models/` — локальные артефакты моделей; файлы моделей не хранятся в Git.
- `results/` — результаты экспериментов.
- `results/tables/` — таблицы с результатами.
- `results/figures/` — графики и иллюстрации.
- `results/logs/` — логи запусков; не хранятся в Git.
- `results/predictions/` — предсказания моделей; не хранятся в Git.
- `docs/` — материалы по плану диссертации, литературе и заметкам.
- `reports/` — черновики отчетов и текстов.

## Workflow

- Код и notebooks хранятся в GitHub.
- Выполнение экспериментов предполагается в Google Colab.
- Данные, модели и результаты хранятся на Google Drive.

## Структура Google Drive

Создание структуры Google Drive нужно выполнить вручную, если папки не были созданы автоматически через подключенный Google Drive.

Нужно создать следующие папки:

```text
MyDrive/russian-dialogue-intent-thesis/
MyDrive/russian-dialogue-intent-thesis/data/raw/
MyDrive/russian-dialogue-intent-thesis/data/interim/
MyDrive/russian-dialogue-intent-thesis/data/processed/
MyDrive/russian-dialogue-intent-thesis/data/annotation/
MyDrive/russian-dialogue-intent-thesis/models/
MyDrive/russian-dialogue-intent-thesis/results/tables/
MyDrive/russian-dialogue-intent-thesis/results/figures/
MyDrive/russian-dialogue-intent-thesis/results/logs/
MyDrive/russian-dialogue-intent-thesis/results/predictions/
MyDrive/russian-dialogue-intent-thesis/exports/
```

## Подключение Google Drive в Colab

```python
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/russian-dialogue-intent-thesis"
```
