# Быстрый старт: Обучение и Предсказание

## 🚀 Команда для запуска обучения

```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"
bash run_training_local.sh
```

## 📋 Что делает команда:

1. Генерирует proto файлы
2. Запускает пассивный сервер обучения (порт 50001)
3. Запускает активный сервер обучения (порт 50000)
4. Запускает клиент обучения
5. Загружает датасеты
6. Обучает модель
7. Сохраняет модель в `example/workdir/active/models/result_model.pkl`

## 🔄 Полный цикл: Обучение → Предсказание

### Шаг 1: Обучение модели
```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"
bash run_training_local.sh
```

### Шаг 2: Предсказание на обученной модели
```bash
bash run_prediction_local.sh
```

## 📝 Альтернативные команды

### Ручной запуск обучения (без скрипта):

```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"

# 1. Генерация proto
bash generate_proto.sh

# 2. Запуск пассивного сервера
python3 python/training_server_passive.py --work_dir=./example/workdir/passive &

# 3. Запуск активного сервера
python3 python/training_server_active.py --work_dir=./example/workdir/active --passive_server_address=localhost:50001 &

# 4. Запуск клиента обучения
cd example
python3 training_client.py \
    --active-dataset=active_dataset_test.csv \
    --passive-dataset=passive_dataset_test.csv \
    --match-id-name=id \
    --label-name=target
```

## ✅ Проверка результатов

После обучения проверьте наличие модели:

```bash
ls -la example/workdir/active/models/result_model.pkl
ls -la example/workdir/passive/models/result_model.pkl
```

## 🎯 Одна команда для всего

```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04 && \
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH" && \
bash run_training_local.sh && \
bash run_prediction_local.sh
```

