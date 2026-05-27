# Артефакты моделей

В эту папку можно положить опциональные артефакты, полученные из основных
ноутбуков ВКР. Если артефактов нет, демо запускается на rule-based fallback.

## Best модель из notebook 11 (по умолчанию)

С notebook 11 (`11_neural_multitask_intent_topic_dialogsum_ru.ipynb`) лучший
итоговый результат на DialogSum-RU дала **single-task RuBERT** модель:

| Метрика | Single-task | Multi-task (λ_topic=0.1 + coarse) |
|---|---|---|
| Test accuracy | **0.9126** | 0.9083 |
| Test macro-F1 | **0.7770** | 0.7580 |
| Test macro-F1 (без `other`) | **0.7914** | — |
| Topic accuracy | — | 0.2130 |
| Coarse topic accuracy | — | 0.5134 |

Вывод: при 10 эпохах single-task модель даёт лучший intent detection;
multi-task полезен как диагностический контекст, но не как основной predict
path. Поэтому webapp по умолчанию использует **single-task** модель.

## Ожидаемые файлы

| Имя файла | Назначение | Обязательность |
|-----------|------------|----------------|
| `single_task_intent_model.pt` | `state_dict` лучшей single-task модели из notebook 11 (encoder + proj + intent_head). | для `single_task_rubert_model` режима |
| `intent_label_encoder.joblib` | `sklearn.preprocessing.LabelEncoder` с классами интентов (порядок индексов соответствует `intent_head`). | для `single_task_rubert_model` режима |
| `multitask_config.json` | Конфиг запуска notebook 11: `model_name`, `max_len`, `num_intents`, классы и пр. Используется для восстановления архитектуры. | желательно |
| `multitask_intent_topic_model.pt` | `state_dict` multi-task модели. | опционально (future extension) |
| `topic_label_encoder.joblib` | LabelEncoder fine-topic классов. | опционально |
| `coarse_topic_label_encoder.joblib` | LabelEncoder coarse-topic классов. | опционально |
| `intent_model.joblib` | Старый sklearn pipeline (legacy). | опционально |
| `topic_centroids.npy` | Матрица центроидов кластеров `(n_clusters, dim)` для sentence-transformers тематики. | опционально |
| `topic_metadata.parquet` | Метаданные кластеров: `cluster_id`, `name`, `description`, `top_words`. | опционально |

> ⚠ Большие `.pt`/`.pickle` файлы **не коммитятся** в git
> (см. корневой `.gitignore`). Эта папка должна остаться placeholder.

## Как подключить из Google Drive (после прогона notebook 11)

Артефакты notebook 11 ожидаются здесь:
`/content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/`

После прогона notebook на Colab скопируйте файлы в `dialogsum_intent_webapp/models/`:

```bash
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/single_task_intent_model.pt dialogsum_intent_webapp/models/
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/intent_label_encoder.joblib dialogsum_intent_webapp/models/
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/multitask_config.json dialogsum_intent_webapp/models/
```

Опционально (только если будете расширять до multi-task):

```bash
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/multitask_intent_topic_model.pt dialogsum_intent_webapp/models/
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/topic_label_encoder.joblib dialogsum_intent_webapp/models/
cp /content/drive/MyDrive/russian-dialogue-intent-thesis/models/multitask_intent_topic/coarse_topic_label_encoder.joblib dialogsum_intent_webapp/models/
```

После копирования перезапустите приложение (`python app.py`). В логе появится
`Обнаружен single-task RuBERT state_dict: single_task_intent_model.pt
(lazy-load при первом predict_intent)`, а в UI верхняя плашка покажет
`Intent mode: single_task_rubert_model`.

## Переменные окружения

| Переменная | Значение по умолчанию | Назначение |
|---|---|---|
| `ENABLE_TORCH_INTENT_MODEL` | `true` | если `false`, single-task RuBERT runtime принудительно отключается |
| `INTENT_MODEL_FILE` | `single_task_intent_model.pt` | имя файла state_dict в `models/` |
| `INTENT_ENCODER_NAME` | (из `multitask_config.json`, иначе `DeepPavlov/rubert-base-cased-conversational`) | имя HuggingFace модели-энкодера |

## Архитектура runtime (для совместимости state_dict)

`SingleTaskIntentModelRuntime` повторяет `SingleTaskIntentModel` из cell 5
notebook 11:

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

State_dict должен быть сохранён из версии notebook 11 после коммита `fbb8252`
(или более актуальной). Если структура архитектуры в будущей версии notebook
поменяется, fallback продолжит работать, но Intent mode будет показывать
`rule_based_fallback`.

## Ограничения fallback

- `intent`: словарь ключевых слов → один из 14 классов; не отличает близкие
  по словам, но разные по смыслу реплики.
- `topic`: 10 предопределённых тем из ВКР, выбирается по совпадению ключевых
  слов; не использует эмбеддинги.
- `summary`: всегда возвращает `None` со статусом
  «Суммаризация не подключена в демо-версии».
