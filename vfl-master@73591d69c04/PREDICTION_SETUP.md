# Инструкция по запуску prediction_client

## ✅ Что исправлено:

1. ✅ Исправлена ошибка в `prediction_client.py` (строка 68: `args.active_features` → `args.passive_features`)
2. ✅ Исправлено использование `ruamel.yaml.safe_load()` → `YAML(typ='safe').load()` в `app.py` и `learning.py`
3. ✅ Восстановлен mock модуль `ucbfl`
4. ✅ Создан скрипт `run_prediction_local.sh` для автоматизации запуска

## 🚀 Запуск prediction_client

### Вариант 1: Использование скрипта (Рекомендуется)

```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"
bash run_prediction_local.sh
```

### Вариант 2: Ручной запуск

1. **Запуск пассивного сервера:**
```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"
python3 python/prediction_server_passive.py --work_dir=./example/workdir/passive &
```

2. **Запуск активного сервера:**
```bash
python3 python/prediction_server_active.py --work_dir=./example/workdir/active --passive_server_address=localhost:50051 &
```

3. **Запуск клиента:**
```bash
cd example
python3 prediction_client.py \
    --active-dataset=active_dataset_test.csv \
    --passive-dataset=passive_dataset_test.csv \
    --match-id-name=id \
    --model-name=result_model.pkl \
    --scores-name=result_scores.csv
```

## ⚠️ Важно: Модель должна быть обучена!

**Ошибка:** `No such file or directory: 'example/workdir/active/models/result_model.pkl'`

**Решение:** Сначала нужно обучить модель с помощью `training_client.py`:

```bash
# 1. Обучить модель
bash run_training_local.sh

# 2. После обучения запустить предсказание
bash run_prediction_local.sh
```

## 📋 Параметры prediction_client

```bash
python3 prediction_client.py \
    --active-address=localhost:50050 \      # Адрес активного сервера
    --active-dataset=active_dataset_test.csv \
    --passive-dataset=passive_dataset_test.csv \
    --match-id-name=id \                    # Столбец для объединения
    --model-name=result_model.pkl \         # Имя файла модели
    --scores-name=result_scores.csv \       # Имя файла результатов
    --active-features=feat_a_00,feat_a_01 \ # Опционально: список признаков
    --passive-features=feat_b_00,feat_b_01 # Опционально: список признаков
```

## 🔄 Полный цикл: Обучение → Предсказание

```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"

# Шаг 1: Обучение модели
bash run_training_local.sh

# Шаг 2: Предсказание на обученной модели
bash run_prediction_local.sh
```

## 📊 Результаты

После успешного выполнения:
- Модель: `example/workdir/active/models/result_model.pkl`
- Результаты предсказания: `example/workdir/active/predict/result_scores.csv`

## 🐛 Устранение проблем

### Проблема: "No such file or directory: result_model.pkl"
**Решение:** Сначала обучите модель с помощью `run_training_local.sh`

### Проблема: "ModuleNotFoundError: No module named 'ucbfl'"
**Решение:** Установите PYTHONPATH: `export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"`

### Проблема: "Connection refused"
**Решение:** Убедитесь, что серверы запущены перед запуском клиента

### Проблема: "safe_load() has been removed"
**Решение:** Уже исправлено в коде. Если появляется - перезапустите серверы.

## ✅ Текущий статус

- ✅ Серверы запускаются
- ✅ Клиент подключается
- ✅ Датасеты загружаются
- ⚠️ Требуется обученная модель для предсказания

