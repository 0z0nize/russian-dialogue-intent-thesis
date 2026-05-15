# Шаблон сводки по EDA (`01_data_loading_and_eda.ipynb`)

Этот шаблон используется для фиксации результатов первичного EDA по датасету
`d0rj/dialogsum-ru` перед началом проектирования схемы интентов.

## Артефакты, которые генерирует ноутбук

Ноутбук `notebooks/01_data_loading_and_eda.ipynb` при запуске в Google Colab
сохраняет следующие артефакты на Google Drive в каталоге
`MyDrive/russian-dialogue-intent-thesis/`.

### Таблицы (`results/tables/`)

- `split_sizes.csv` — размеры сплитов `train` / `validation` / `test`.
- `dialogue_length_stats.csv` — описательная статистика по длинам диалогов
  (число реплик, длина в символах, длина в словах) по сплитам.
- `summary_length_stats.csv` — описательная статистика по длинам резюме
  (в символах и словах) по сплитам.
- `top20_first_tokens.csv` — топ-20 наиболее частых первых токенов реплик.

### Графики (`results/figures/`)

- `hist_num_utterances.png` — гистограмма числа реплик в диалоге.
- `hist_dialogue_len_words.png` — гистограмма длины диалога в словах.
- `hist_summary_len_words.png` — гистограмма длины резюме в словах.
- `box_dialogue_len_words_by_split.png` — boxplot длины диалога по сплитам.
- `box_num_utterances_by_split.png` — boxplot числа реплик по сплитам.

## Файлы, которые должны быть на Google Drive после запуска

```text
MyDrive/russian-dialogue-intent-thesis/results/tables/split_sizes.csv
MyDrive/russian-dialogue-intent-thesis/results/tables/dialogue_length_stats.csv
MyDrive/russian-dialogue-intent-thesis/results/tables/summary_length_stats.csv
MyDrive/russian-dialogue-intent-thesis/results/tables/top20_first_tokens.csv
MyDrive/russian-dialogue-intent-thesis/results/figures/hist_num_utterances.png
MyDrive/russian-dialogue-intent-thesis/results/figures/hist_dialogue_len_words.png
MyDrive/russian-dialogue-intent-thesis/results/figures/hist_summary_len_words.png
MyDrive/russian-dialogue-intent-thesis/results/figures/box_dialogue_len_words_by_split.png
MyDrive/russian-dialogue-intent-thesis/results/figures/box_num_utterances_by_split.png
```

## Вопросы, на которые нужно ответить после EDA, до разметки интентов

- Какой реальный объём данных в `train`/`validation`/`test` и хватит ли его для
  выбранного подхода к классификации интентов?
- Какова типичная длина диалога (число реплик, число слов) и как это влияет на
  выбор модели и max sequence length?
- Однороден ли формат поля `dialogue` (строка с переносами, список реплик, иное)
  и нужна ли отдельная нормализация перед разметкой?
- Какие темы и сценарии встречаются в выборке (по `topic` и резюме) и можно ли
  на их основе предварительно очертить набор кандидатных интентов?
- Содержат ли диалоги несколько намерений одновременно — стоит ли формулировать
  задачу как single-label или multi-label?
- Нужно ли ограничить разметку подвыборкой (например, по длине или по теме) для
  пилотной разметки, и каков критерий отбора?
- Есть ли в данных явный шум / артефакты перевода, требующие предварительной
  фильтрации?
- Какой baseline считать референсным (например, перевод диалогов на английский
  + англоязычная intent-модель) и какие метрики фиксировать?
