# Артефакты моделей

В эту папку можно положить опциональные артефакты, полученные из основных
ноутбуков ВКР. Если артефактов нет ни здесь, ни в публичном Hugging Face
репозитории — демо запускается на rule-based fallback.

## Порядок поиска артефактов

`model_pipeline.load_artifacts()` ищет файлы в следующем порядке:

1. локальная папка `dialogsum_intent_webapp/models/` (этот каталог);
2. Hugging Face Hub: `$HF_INTENT_REPO_ID`
   (по умолчанию `ozonize/dialogsum-ru-intent-rubert`).

В UI и в `get_artifact_status()` источник отображается как
`torch_intent_state_source = "local" | "huggingface_hub" | "missing"`.

## Best модель (по умолчанию)

Лучший итоговый результат на DialogSum-RU дала **single-task RuBERT** модель.
Полные текстовые дубликаты реплик выявлены и устранены до разбиения на
train/val/test (удалено 425 из 4 722, итоговый корпус 4 297; train 3 007 /
val 645 / test 645); повторная оценка подтвердила устойчивость модели:

| Метрика | Single-task | Multi-task (λ_topic=0.1 + coarse) |
|---|---|---|
| Test accuracy | **0.9054** | 0.8930 |
| Test macro-F1 | **0.7794** | 0.7612 |
| Test macro-F1 (без `other`) | **0.8066** | 0.7813 |
| Test weighted-F1 | **0.9034** | 0.8889 |
| Topic accuracy | — | 0.1705 |
| Coarse topic accuracy | — | 0.5178 |

Вывод: при 10 эпохах single-task модель даёт лучший intent detection;
multi-task полезен как диагностический контекст, но не как основной predict
path. Поэтому webapp по умолчанию использует **single-task** модель.

## Ожидаемые файлы

| Имя файла | Назначение | Обязательность |
|-----------|------------|----------------|
| `single_task_intent_model.pt` | `state_dict` лучшей single-task модели (encoder + proj + intent_head). | для `single_task_rubert_model` режима |
| `intent_label_encoder.joblib` | `sklearn.preprocessing.LabelEncoder` с классами интентов (порядок индексов соответствует `intent_head`). | для `single_task_rubert_model` режима |
| `multitask_config.json` | Конфиг архитектуры: `model_name`, `max_len`, `num_intents`, классы и пр. | желательно |
| `multitask_intent_topic_model.pt` | `state_dict` multi-task модели. | опционально (future extension) |
| `topic_label_encoder.joblib` | LabelEncoder fine-topic классов. | опционально |
| `coarse_topic_label_encoder.joblib` | LabelEncoder coarse-topic классов. | опционально |
| `intent_model.joblib` | Старый sklearn pipeline (legacy). | опционально |
| `topic_centroids.npy` | Матрица центроидов кластеров `(n_clusters, dim)` для sentence-transformers тематики. | опционально |
| `topic_metadata.parquet` | Метаданные кластеров: `cluster_id`, `name`, `description`, `top_words`. | опционально |
| `summarizer/` или `summarization/` | Папка с весами seq2seq суммаризатора (`config.json`, `pytorch_model.bin` / `model.safetensors`, токенизатор). Загружается через `AutoModelForSeq2SeqLM.from_pretrained(<path>)`. Если папки нет, используется HuggingFace Hub (`SUMMARIZATION_MODEL_NAME`, по умолчанию `IlyaGusev/rut5_base_sum_gazeta`). | опционально |

> ⚠ Большие `.pt`/`.pickle` файлы **не коммитятся** в git
> (см. корневой `.gitignore`). Эта папка должна остаться placeholder.

## Как подключить из Hugging Face Hub

Артефакты публикуются в репозитории
[`ozonize/dialogsum-ru-intent-rubert`](https://huggingface.co/ozonize/dialogsum-ru-intent-rubert).
При первом запуске webapp `load_artifacts()` сам скачает недостающие файлы
через `huggingface_hub.hf_hub_download` в локальный кэш
(`~/.cache/huggingface` или `$HF_HOME`). Копировать вручную не обязательно.

Если требуется заранее положить файлы локально (например, при работе без
сети), используйте `huggingface-cli`:

```bash
huggingface-cli download ozonize/dialogsum-ru-intent-rubert \
    single_task_intent_model.pt intent_label_encoder.joblib multitask_config.json \
    --local-dir dialogsum_intent_webapp/models
```

Опционально (только если будете расширять до multi-task):

```bash
huggingface-cli download ozonize/dialogsum-ru-intent-rubert \
    multitask_intent_topic_model.pt topic_label_encoder.joblib coarse_topic_label_encoder.joblib \
    --local-dir dialogsum_intent_webapp/models
```

После этого `python app.py` поднимет интерфейс в режиме
`single_task_rubert_model`. В логе появится
`Обнаружен single-task RuBERT state_dict: single_task_intent_model.pt
(lazy-load при первом predict_intent)`, а в UI верхняя плашка покажет
`Intent mode: single_task_rubert_model`.

## Переменные окружения

| Переменная | Значение по умолчанию | Назначение |
|---|---|---|
| `ENABLE_TORCH_INTENT_MODEL` | `true` | если `false`, single-task RuBERT runtime принудительно отключается |
| `INTENT_MODEL_FILE` | `single_task_intent_model.pt` | имя файла state_dict в `models/` и в HF-репозитории |
| `INTENT_ENCODER_NAME` | (из `multitask_config.json`, иначе `DeepPavlov/rubert-base-cased-conversational`) | имя HuggingFace модели-энкодера |
| `HF_INTENT_REPO_ID` | `ozonize/dialogsum-ru-intent-rubert` | HF-репозиторий с обученными артефактами |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | — | токен Hugging Face, нужен только для приватных репозиториев |
| `ENABLE_HF_DOWNLOAD` | `true` | если `false`, загрузка с HF Hub отключена (только локальные файлы) |
| `ENABLE_SUMMARIZATION` | `true` | если `false`, summarize() возвращает статус «отключено» |
| `SUMMARIZATION_MODEL_NAME` | `IlyaGusev/rut5_base_sum_gazeta` | HF id, fallback если нет локальной папки summarizer |
| `SUMMARIZATION_LOCAL_DIR` | — | явный путь к локальной папке summarizer (с `config.json`) |
| `SUMMARIZATION_MAX_INPUT_TOKENS` / `SUMMARIZATION_MAX_NEW_TOKENS` / `SUMMARIZATION_NUM_BEAMS` | `1024 / 96 / 4` | параметры генерации |

## Архитектура runtime (для совместимости state_dict)

`SingleTaskIntentModelRuntime` повторяет архитектуру `SingleTaskIntentModel`:

```
encoder       = AutoModel.from_pretrained(model_name)
proj          = Sequential(Linear(h, h), GELU(), LayerNorm(h), Dropout(p))
intent_head   = Linear(h, num_intents)
forward(pooled) = intent_head(proj(mean_pool(encoder.last_hidden_state)))
```

Ключи state_dict: `encoder.*`, `proj.*`, `intent_head.*`. При загрузке runtime
сначала пробует `strict=True`, при несовпадении переходит в `strict=False` и
пишет предупреждение в `artifact_status.torch_intent_load_error`. Чужие
головы (например, `topic_head.*` из multi-task чекпойнта) автоматически
отфильтровываются. Если ключи `encoder./proj./intent_head.` вообще отсутствуют,
runtime не падает: webapp молча переходит в rule-based fallback и сообщает об
этом в `artifact_status`.

## Ограничения fallback

- `intent`: словарь ключевых слов → один из 14 классов; не отличает близкие
  по словам, но разные по смыслу реплики.
- `topic`: 10 предопределённых тем из ВКР, выбирается по совпадению ключевых
  слов; не использует эмбеддинги.
- `summary`: если `transformers`/`torch` не установлены или загрузка модели
  упала, возвращается `None` со статусом
  `Суммаризация недоступна: <причина>`. По умолчанию суммаризация включена
  через HuggingFace seq2seq (см. секцию «Суммаризация» в основном README).
