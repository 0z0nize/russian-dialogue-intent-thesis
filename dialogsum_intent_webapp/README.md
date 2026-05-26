# DialogSum-RU intent demo · Gradio webapp

Демонстрационный веб-интерфейс к магистерской ВКР
**«Семантический анализ русскоязычных диалогов для задачи распознавания
намерений с улучшением на базе предобученных моделей»** (НИУ ИТМО,
магистратура «Аналитика данных»).

Приложение принимает на вход русскоязычный текст или голосовую запись и
возвращает:

- intent (намерение из 14 классов речевых актов, описанных в ВКР);
- тематический кластер (id, название, описание, top-words);
- статус суммаризации (заглушка в текущей демо-версии);
- полный JSON-ответ для отладки.

## Архитектура

```
audio ──► faster-whisper (lazy)
                │
                ▼
            текст ──► intent (joblib model OR rule-based fallback)
                  │
                  └─► topic (sentence-transformers + centroids OR keyword fallback)
                  │
                  └─► summary (заглушка)
```

- UI: **Gradio Blocks** (две вкладки «Текст» / «Голос»), русский интерфейс.
- STT: **faster-whisper** (русский язык, ленивая загрузка модели).
- Intent / topic: опциональные артефакты из ноутбуков ВКР; при отсутствии
  включаются rule-based fallback'и.
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

## Подключение реальных артефактов

Положите в `models/` следующие файлы (любое подмножество):

| Файл | Назначение |
|------|------------|
| `intent_model.joblib` | sklearn-совместимая модель / pipeline для intent. |
| `intent_label_encoder.joblib` | `LabelEncoder` для 14 классов из ВКР. |
| `topic_centroids.npy` | Матрица центроидов кластеров, shape `(n_clusters, dim)`. |
| `topic_metadata.parquet` | Метаданные: `cluster_id`, `name`, `description`, `top_words`. |

Подробнее — см. `models/README.md`.

## Ограничения mock fallback

- Intent fallback — по ключевым словам, набор покрывает базовые классы
  (greeting, thanks, farewell, информационный/сервисный запрос, бронирование,
  жалоба, problem report, договорённость, подтверждение, отказ, рекомендация,
  мнение и `other`). На сложных репликах точность ограничена.
- Topic fallback — 10 заранее заданных тем из ВКР: развлечения, музыкальные
  события, дом, жалоба, образование, поиск книг, путешествия, работа,
  собеседование, ремонт/обслуживание. Выбор по совпадению ключевых слов.
- Summary — всегда `None` со статусом
  «Суммаризация не подключена в демо-версии».

## API analyze()

```python
from model_pipeline import analyze
analyze("Здравствуйте, я хочу забронировать билет")
# {
#   "input_text": ...,
#   "intent_label": "purchase_or_booking_request",
#   "intent_confidence": 0.6,
#   "intent_mode": "rule_based_fallback",
#   "topic_cluster_id": 6,
#   "topic_cluster_name": "путешествия",
#   ...
#   "summary": None,
#   "summary_status": "Суммаризация не подключена в демо-версии",
# }
```
