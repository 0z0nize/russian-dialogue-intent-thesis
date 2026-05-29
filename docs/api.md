# HTTP API публичного демо `intent-demo.online`

Этот документ описывает HTTP API публичного демонстрационного веб-приложения ВКР
**«Семантический анализ русскоязычных диалогов для задачи распознавания
намерений с улучшением на базе предобученных моделей»**.

Веб-приложение реализовано на [Gradio](https://www.gradio.app/) (см.
[`dialogsum_intent_webapp/app.py`](../dialogsum_intent_webapp/app.py)), поэтому
API представляет собой стандартный **Gradio HTTP API** —
двухшаговый протокол `POST → event_id → GET → результат`.

- **Base URL (production):** <https://intent-demo.online>
- **Base URL (локальный запуск):** `http://localhost:7860`
- **Префикс API:** `/gradio_api`
- **Формат:** `application/json` для большинства запросов; `multipart/form-data`
  для загрузки аудио-файлов.
- **Аутентификация:** отсутствует, API публичный.
- **Postman-коллекция:** [`docs/postman/intent_demo_postman_collection.json`](postman/intent_demo_postman_collection.json) —
  готовая коллекция со всеми запросами и автозаполнением `event_id` в
  переменную коллекции через post-response script.

> ⚠ Это **демонстрационный** API, привязанный к Gradio-приложению. Имена
> функций (`analyze_text`, `analyze_audio`, …) и их сигнатуры могут меняться
> вместе с UI. Для долгосрочной интеграции рекомендуется поднимать собственный
> экземпляр приложения (см. [`dialogsum_intent_webapp/README.md`](../dialogsum_intent_webapp/README.md))
> и фиксировать версию кода.

## Общая схема вызова Gradio API

Каждый функциональный эндпоинт состоит из **двух шагов**:

1. **POST** на `/{base_url}/gradio_api/call/<endpoint_name>` с JSON-телом
   `{"data": [...]}` — возвращает идентификатор события `event_id`.
2. **GET** на `/{base_url}/gradio_api/call/<endpoint_name>/<event_id>` —
   возвращает результат в формате [Server-Sent Events](https://developer.mozilla.org/ru/docs/Web/API/Server-sent_events)
   (`event: complete` / `data: [...]`).

### Минимальный пример (curl)

```bash
# 1) POST — отправили входные данные, получили event_id
EVENT_ID=$(curl -s -X POST https://intent-demo.online/gradio_api/call/analyze_text \
  -H "Content-Type: application/json" \
  -d '{"data": ["Здравствуйте, я хочу оформить возврат товара"]}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['event_id'])")

# 2) GET — забрали результат по event_id (SSE)
curl -N https://intent-demo.online/gradio_api/call/analyze_text/$EVENT_ID
```

Ответ GET-шага — поток SSE-событий, последнее сообщение содержит JSON-массив
с результатами (порядок элементов соответствует выходам Gradio-компонентов
данной функции — см. ниже разделы для каждого эндпоинта).

### Минимальный пример (Python, `gradio_client`)

Для скриптовой работы удобнее официальная библиотека
[`gradio_client`](https://www.gradio.app/docs/python-client/introduction), она
скрывает двухшаговый протокол:

```python
from gradio_client import Client

client = Client("https://intent-demo.online")

# api_name можно подсмотреть через client.view_api() или взять из этого файла
result = client.predict(
    "Здравствуйте, я хочу оформить возврат товара",
    api_name="/analyze_text",
)
print(result)
```

## Эндпоинты

Ниже перечислены все публично доступные функции Gradio API.

> Имена эндпоинтов с префиксом `lambda*` / `_clear_voice_outputs` —
> внутренние UI-обработчики (очистка полей, переключение режимов и т. п.).
> Они оставлены в коллекции для полноты, но в обычной интеграции не нужны.

### 1. `analyze_text` — анализ одного фрагмента текста

Возвращает intent, тематический кластер и (опционально) сводку для одного
произвольного куска русскоязычного текста. Текст обрабатывается **как единое
целое** — без разбиения на реплики. Подходит для коротких высказываний.

**Эндпоинт:** `POST /gradio_api/call/analyze_text`

**Тело запроса:**

```json
{
  "data": ["Здравствуйте, я хочу оформить возврат товара"]
}
```

| Поле `data[i]` | Тип | Описание |
| --- | --- | --- |
| `data[0]` | `string` | Произвольный русскоязычный текст. |

**Шаг GET:** `GET /gradio_api/call/analyze_text/<event_id>`

**Структура результата** (массив выходов, порядок соответствует UI):

| Индекс | Поле UI | Содержимое |
| --- | --- | --- |
| 0 | `Интент` | Метка интента (см. таблицу классов ниже). |
| 1 | `Уверенность intent` | `float` ∈ [0, 1]. |
| 2 | `Intent mode` | `single_task_rubert_model` / `sklearn_intent_model` / `rule_based_fallback`. |
| 3 | `Topic cluster id` | `int`, `-1` если кластер не определён. |
| 4 | `Topic cluster name` | Человекочитаемое название кластера. |
| 5 | `Topic cluster description` | Описание кластера / диагностика. |
| 6 | `Top words` | Топ-слова кластера. |
| 7 | `Сводка` | Сгенерированная сводка (`null`, если суммаризация выключена / упала). |
| 8 | `Полный JSON` | Полный словарь `analyze()` (см. [`model_pipeline.py`](../dialogsum_intent_webapp/model_pipeline.py)). |

### 2. `analyze_text_utterances` — по-репличный анализ диалога

Парсит вход на отдельные реплики по маркерам спикеров (DialogSum-формат
`#Person1#:` / `#Person2#:`, а также русские варианты `Говорящий 1`,
`Спикер 2`, `Собеседник 1`) и предсказывает intent для каждой реплики
отдельно. Рекомендуется для длинных диалогов.

**Эндпоинт:** `POST /gradio_api/call/analyze_text_utterances`

**Тело запроса:**

```json
{
  "data": [
    "#Person1#: Здравствуйте, у меня не работает интернет.\n#Person2#: Подскажите, пожалуйста, ваш номер договора."
  ]
}
```

**Шаг GET:** `GET /gradio_api/call/analyze_text_utterances/<event_id>`

**Структура результата:**

| Индекс | Содержимое |
| --- | --- |
| 0 | Таблица `[[№, Говорящий, Реплика, Интент, Уверенность, Top-3], ...]`. |
| 1 | Полный список `items` от `model_pipeline.analyze_utterances()`. |

### 3. `analyze_audio` — STT + полный пайплайн

Принимает аудио-файл, прогоняет его через `faster-whisper`
(русская модель, ленивая загрузка), а затем анализирует распознанный текст
как `analyze_text`.

**Эндпоинт:** `POST /gradio_api/call/analyze_audio`

**Формат входа:** Gradio ожидает в `data[0]` объект
[`gradio.FileData`](https://www.gradio.app/docs/gradio/filedata) — либо
ссылку на файл по HTTP(S), либо путь, ранее полученный через
`/gradio_api/upload` (см. ниже):

```json
{
  "data": [
    {
      "path": "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
      "meta": {"_type": "gradio.FileData"}
    }
  ]
}
```

**Шаг GET:** `GET /gradio_api/call/analyze_audio/<event_id>`

**Структура результата:** то же, что у `analyze_text`, плюс дополнительное
первое поле с распознанным текстом:

| Индекс | Содержимое |
| --- | --- |
| 0 | Распознанный текст (Whisper). |
| 1–9 | Те же поля, что у `analyze_text` (intent, confidence, …, полный JSON). |

> ⚠ Запись с микрофона в браузере доступна **только по HTTPS** или с
> `localhost`. Для серверного использования передавайте аудио файлом
> (см. следующий раздел).

### 4. `/gradio_api/upload` — загрузка локального аудио-файла

Если у вас есть локальный файл (а не URL), его сначала нужно загрузить на
сервер Gradio, чтобы получить путь, который потом можно подставить в
`analyze_audio`.

**Эндпоинт:** `POST /gradio_api/upload`

**Формат:** `multipart/form-data`, поле `files` — файл.

```bash
curl -X POST https://intent-demo.online/gradio_api/upload \
  -F "files=@/path/to/local/audio.wav"
```

Ответ — JSON-массив со строкой пути вида
`/tmp/gradio/...../audio.wav`, который нужно подставить в `path` шага
`analyze_audio` (с тем же `meta: {"_type": "gradio.FileData"}`).

### 5. Служебные эндпоинты UI

Перечисленные ниже эндпоинты — это привязанные к кнопкам очистки и
переключения UI-обработчики Gradio. В обычной интеграции они не нужны,
но они присутствуют в Postman-коллекции для полноты.

| Эндпоинт | Назначение |
| --- | --- |
| `lambda`, `lambda_1` … `lambda_4` | Анонимные UI-callback'и (показ/скрытие блоков, обновление надписей). Возвращают служебные UI-объекты. |
| `_clear_voice_outputs` | Сбрасывает поля результатов на вкладке «Голос». |

## Классы интента

Финальная модель — single-task RuBERT на базе
`DeepPavlov/rubert-base-cased-conversational`, обучена на 14 классах
речевых актов. Полный список и определения — глава 6 ВКР и
[`dialogsum_intent_webapp/README.md`](../dialogsum_intent_webapp/README.md).

Краткое перечисление меток, которые возвращает API (поле `intent_label`):

```
greeting, farewell, thanks,
information_request, service_request, purchase_or_booking_request,
problem_report, complaint, agreement, confirmation, refusal,
recommendation, opinion, other
```

## Коды ответа и ошибки

- `200 OK` — успешный шаг POST (вернулся `event_id`) или успешный SSE-поток
  GET.
- `4xx` / `5xx` — Gradio возвращает ошибку в SSE-событии
  `event: error` либо в JSON-теле; в `intent_mode` / `summary_status`
  возвращаемого результата отражается тип ошибки (`error: ...`).
- Если артефакты intent-модели не подгружены, API всё равно ответит, но
  поле `intent_mode` будет `rule_based_fallback`, а уверенность — низкой.
- Лимиты по размеру файла зависят от Nginx (`client_max_body_size`) и
  от ресурсов VM, на которой развёрнуто демо. Для аудио рекомендуется
  не превышать ~10 МБ.

## Совместимость и стабильность

API публичного демо предоставляется **как есть** для целей демонстрации
ВКР. Стабильность сигнатур не гарантируется между релизами веб-приложения.
Для воспроизводимых экспериментов:

- зафиксируйте коммит в [репозитории](https://github.com/0z0nize/russian-dialogue-intent-thesis)
  и поднимите собственный экземпляр приложения по инструкции
  [`dialogsum_intent_webapp/README.md`](../dialogsum_intent_webapp/README.md);
- либо используйте библиотеку
  [`gradio_client`](https://www.gradio.app/docs/python-client/introduction)
  и метод `Client.view_api()` для динамического обнаружения сигнатур.

## См. также

- [Postman-коллекция](postman/intent_demo_postman_collection.json) — готовые
  запросы со всеми эндпоинтами и автоматическим прокидыванием `event_id`.
- [`dialogsum_intent_webapp/README.md`](../dialogsum_intent_webapp/README.md) —
  архитектура веб-приложения, переменные окружения, развёртывание,
  описание `model_pipeline.analyze()`.
- [`README.md`](../README.md) — общий обзор ВКР и репозитория.
