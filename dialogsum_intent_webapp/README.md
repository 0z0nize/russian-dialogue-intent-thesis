# DialogSum-RU intent demo · Gradio webapp

Демонстрационный веб-интерфейс к магистерской ВКР
**«Семантический анализ русскоязычных диалогов для задачи распознавания
намерений с улучшением на базе предобученных моделей»** (НИУ ИТМО,
магистратура «Аналитика данных»).

Приложение принимает на вход русскоязычный текст или голосовую запись и
возвращает:

- intent (намерение из 14 классов речевых актов, описанных в ВКР);
- тематический кластер (id, название, описание, top-words);
- сводку (HuggingFace seq2seq, по умолчанию `IlyaGusev/rut5_base_sum_gazeta`);
- полный JSON-ответ для отладки.

## Архитектура

```
audio ──► faster-whisper (lazy)
                │
                ▼
            текст ──► intent
                   │     ├── single-task RuBERT runtime (notebook 11, BEST) — по умолчанию
                   │     ├── sklearn intent_model.joblib (legacy)
                   │     └── rule-based fallback
                   │
                   ├─► topic (sentence-transformers + centroids OR keyword fallback)
                   │
                   └─► summary (HuggingFace seq2seq, lazy-load)
```

- UI: **Gradio Blocks** (две вкладки «Текст» / «Голос»), русский интерфейс.
  Сверху на странице видна плашка с текущим `Intent mode`
  (`single_task_rubert_model` / `sklearn_intent_model` / `rule_based_fallback`),
  тот же режим дублируется в результатах под каждой репликой.
- STT: **faster-whisper** (русский язык, ленивая загрузка модели).
- Intent: лучшая модель — single-task RuBERT из ноутбука 11
  (`DeepPavlov/rubert-base-cased-conversational`, accuracy 0.9126, macro-F1 0.7770).
  При отсутствии артефактов включается rule-based fallback.
- Topic: пока остаётся в режиме centroids/keyword fallback —
  multi-task topic head из ноутбука 11 даёт topic accuracy ~0.21 (fine) и
  ~0.51 (coarse), что хуже отдельной кластерной пайплайны.
- Без платных внешних API.

## Структура проекта

```
dialogsum_intent_webapp/
├── app.py                    # Gradio UI
├── model_pipeline.py         # STT + intent + topic + summary
├── requirements.txt
├── README.md                 # вы здесь
├── models/                   # опциональные артефакты
│   ├── .gitkeep
│   └── README.md
└── deploy/
    ├── Dockerfile
    └── docker-compose.yml
```

## Локальный запуск

Требуется Python 3.10+ и `ffmpeg` (для STT).

```bash
cd dialogsum_intent_webapp
pip install -r requirements.txt
python app.py
```

Откроется на `http://127.0.0.1:7860`.

## Запуск на VPS по публичному IP

```bash
sudo ufw allow 7860/tcp
cd dialogsum_intent_webapp
python app.py
```

Затем открыть `http://<SERVER_PUBLIC_IP>:7860`.

> Примечание: запись с микрофона в браузере обычно требует HTTPS-контекста.
> На голом IP надёжнее использовать **upload** аудиофайла. Для production
> рекомендуется домен + SSL/Nginx-reverse-proxy перед Gradio.

## Запуск через Docker

```bash
cd dialogsum_intent_webapp/deploy
docker compose up --build -d
```

Открыть `http://<SERVER_PUBLIC_IP>:7860`.

Переменные окружения (см. `docker-compose.yml`):

- `WHISPER_MODEL_SIZE` — размер Whisper-модели: `tiny | base | small | medium | large-v3`.
  По умолчанию в compose — `small` (компромисс для слабого VPS). В коде по
  умолчанию `medium`.
- `SENTENCE_MODEL` — модель эмбеддингов для тематики
  (по умолчанию `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- `MODELS_DIR` — путь к артефактам внутри контейнера (`/app/models`).

Папка `models/` хоста монтируется в контейнер, так что артефакты можно
обновлять без пересборки образа.

## Подключение лучшей модели из notebook 11

Notebook `11_neural_multitask_intent_topic_dialogsum_ru.ipynb` сохраняет
артефакты в Google Drive:
`/content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/`

Лучший по итогам последнего прогона режим — **single-task RuBERT**
(`DeepPavlov/rubert-base-cased-conversational`):

- Test accuracy **0.9126**
- Test macro-F1 **0.7770**
- Test macro-F1 без `other` **0.7914**
- Сравнение: multi-task `lambda_topic=0.1 + coarse` даёт accuracy 0.9083 /
  macro-F1 0.7580, topic acc 0.2130, coarse topic acc 0.5134.

### Автоматическое подключение артефактов из Google Drive

`model_pipeline.py` ищет файлы intent-артефактов в следующем порядке:

1. локальная папка `dialogsum_intent_webapp/models/`;
2. Google Drive: `$DRIVE_MULTITASK_MODELS_DIR`
   (по умолчанию `$PROJECT_DRIVE_DIR/models/multitask_intent_topic`,
   то есть `/content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/`).

Если хотя бы один путь существует, артефакты подхватываются автоматически —
копировать файлы в локальную `models/` не обязательно. В UI верхняя плашка
показывает источник (`local` / `google_drive`) и состояние Drive.

**В Colab:** перед запуском webapp смонтируйте Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Локально / на VPS:** Drive не смонтирован, поэтому либо положите файлы в
локальную `models/`, либо переопределите путь через переменные окружения:

```bash
export PROJECT_DRIVE_DIR=/mnt/drive/russian-dialogue-intent-thesis
# или сразу конкретные подпапки:
export DRIVE_MULTITASK_MODELS_DIR=/mnt/models/multitask_intent_topic
```

Альтернатива — скопировать руками:

```bash
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/single_task_intent_model.pt dialogsum_intent_webapp/models/
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/intent_label_encoder.joblib dialogsum_intent_webapp/models/
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/multitask_config.json dialogsum_intent_webapp/models/
```

После этого `python app.py` поднимет интерфейс в режиме
`single_task_rubert_model`. Сам HuggingFace-энкодер (~700 МБ) скачивается
лениво при первом запросе, поэтому старт UI остаётся быстрым.

> ⚠ Большие `.pt`/`.bin`/`.pickle` артефакты **не коммитятся** в git
> (корневой `.gitignore`). Папка `models/` в репозитории остаётся
> placeholder с `.gitkeep` и `README.md`.

## Суммаризация

Суммаризация **включена по умолчанию** (`ENABLE_SUMMARIZATION=true`) и
выполняется через HuggingFace `AutoModelForSeq2SeqLM`. Модель грузится
лениво при первом обращении к `summarize()`/`analyze()`.

Порядок поиска весов суммаризатора:

1. локальная папка с `config.json`: `$SUMMARIZATION_LOCAL_DIR` →
   `models/summarizer` → `models/summarization`;
2. Google Drive: `$DRIVE_SUMMARIZATION_DIR` →
   `$PROJECT_DRIVE_DIR/models/summarization` →
   `$PROJECT_DRIVE_DIR/models/summarizer`;
3. HuggingFace Hub: `$SUMMARIZATION_MODEL_NAME` (по умолчанию
   `IlyaGusev/rut5_base_sum_gazeta`).

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `ENABLE_SUMMARIZATION` | `true` | если `false`, статус будет «Суммаризация отключена» |
| `SUMMARIZATION_MODEL_NAME` | `IlyaGusev/rut5_base_sum_gazeta` | HF id, используется если локально / в Drive модели нет |
| `SUMMARIZATION_LOCAL_DIR` | — | явный путь к локальной папке summarizer (с `config.json`) |
| `DRIVE_SUMMARIZATION_DIR` | — | явный путь к папке summarizer на Drive |
| `SUMMARIZATION_MAX_INPUT_TOKENS` | `1024` | обрезка входа по токенам |
| `SUMMARIZATION_MAX_NEW_TOKENS` | `96` | длина генерации |
| `SUMMARIZATION_NUM_BEAMS` | `4` | beam search |

> ⚠ Первый вызов может скачивать веса с HuggingFace (rut5-base ≈ 850 MB) и
> занимать минуту-другую. GPU рекомендуется, на CPU суммаризация работает,
> но заметно медленнее.

### Переменные окружения для intent-модели

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `ENABLE_TORCH_INTENT_MODEL` | `true` | если `false`, single-task RuBERT runtime отключён принудительно |
| `INTENT_MODEL_FILE` | `single_task_intent_model.pt` | имя файла state_dict в `models/` |
| `INTENT_ENCODER_NAME` | (из `multitask_config.json`, иначе `DeepPavlov/rubert-base-cased-conversational`) | имя HuggingFace модели-энкодера |
| `MODELS_DIR` | `models` | путь к локальным артефактам |
| `PROJECT_DRIVE_DIR` | `/content/drive/MyDrive/russian-dialogue-intent-thesis` | корень проекта на Google Drive |
| `DRIVE_MULTITASK_MODELS_DIR` | `$PROJECT_DRIVE_DIR/models/multitask_intent_topic` | папка с artefacts ноутбука 11 на Drive |

### Дополнительные опциональные артефакты

| Файл | Назначение |
|------|------------|
| `intent_model.joblib` | старый sklearn-совместимый pipeline для intent (legacy fallback). |
| `topic_centroids.npy` | матрица центроидов кластеров, shape `(n_clusters, dim)`. |
| `topic_metadata.parquet` | метаданные: `cluster_id`, `name`, `description`, `top_words`. |
| `multitask_intent_topic_model.pt` | state_dict multi-task модели — пока **не используется** на runtime; зарезервировано как future extension. |

Подробнее — см. [`models/README.md`](models/README.md).

### Что показывает UI

Плашка вверху страницы (и поле «Intent mode» в результатах) явно
сигнализирует, в каком режиме сработал intent-классификатор:

- `single_task_rubert_model` — загружены `single_task_intent_model.pt` +
  `intent_label_encoder.joblib`, прогноз сделан RuBERT-моделью.
- `sklearn_intent_model` — single-task артефактов нет, использован старый
  `intent_model.joblib`.
- `rule_based_fallback` — артефактов нет, сработали ключевые слова.

## Ограничения mock fallback

- Intent fallback — по ключевым словам, набор покрывает базовые классы
  (greeting, thanks, farewell, информационный/сервисный запрос, бронирование,
  жалоба, problem report, договорённость, подтверждение, отказ, рекомендация,
  мнение и `other`). На сложных репликах точность ограничена.
- Topic fallback — 10 заранее заданных тем из ВКР: развлечения, музыкальные
  события, дом, жалоба, образование, поиск книг, путешествия, работа,
  собеседование, ремонт/обслуживание. Выбор по совпадению ключевых слов.
- Summary — если `transformers`/`torch` недоступны или модель не удалось
  скачать, возвращается `None` со статусом
  `Суммаризация недоступна: <причина>`. Через
  `ENABLE_SUMMARIZATION=false` суммаризацию можно явно отключить.

## API analyze()

```python
from model_pipeline import analyze, get_artifact_status

get_artifact_status()
# {
#   "torch_intent_state_found": True / False,
#   "intent_label_encoder_found": True / False,
#   "multitask_config_found": True / False,
#   "intent_mode_planned": "single_task_rubert_model" | "rule_based_fallback",
#   ...
# }

analyze("Здравствуйте, я хочу забронировать билет")
# {
#   "input_text": ...,
#   "intent_label": "purchase_or_booking_request",
#   "intent_confidence": 0.93,
#   "intent_mode": "single_task_rubert_model",  # или rule_based_fallback
#   "intent_topk": [{"label": ..., "confidence": ...}, ...],
#   "topic_cluster_id": 6,
#   "topic_cluster_name": "путешествия",
#   ...
#   "summary": "...",
#   "summary_status": "Суммаризация выполнена (huggingface_hub): IlyaGusev/rut5_base_sum_gazeta",
#   "artifact_status": {
#       "drive_available": True / False,
#       "torch_intent_state_source": "local" | "google_drive" | "missing",
#       "summary_mode": "transformers_seq2seq" | "disabled" | "error" | "pending_lazy_load",
#       "summarizer_source": "local" | "google_drive" | "huggingface_hub",
#       ...
#   },
# }
```
