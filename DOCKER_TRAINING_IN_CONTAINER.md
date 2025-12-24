# Запуск обучения в Docker контейнере

## 🐳 Шаг 1: Пересборка и перезапуск контейнера

```bash
# На хосте (если контейнер запущен - выйдите: exit)
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection

# Остановка и удаление старого контейнера
docker-compose down

# Пересборка образа (с новыми зависимостями)
docker-compose build

# Запуск контейнера
docker-compose up -d

# Вход в контейнер
docker-compose exec vfl-feature-selection bash
```

## 🚀 Шаг 2: Запуск обучения внутри контейнера

```bash
# Внутри контейнера (root@...:/app#)
cd /app/vfl-master@73591d69c04
export PYTHONPATH="/app/ucbfl_mock:$PYTHONPATH"
bash run_training_local.sh
```

## 📋 Полная последовательность команд

### На хосте:
```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection
docker-compose down
docker-compose build
docker-compose up -d
docker-compose exec vfl-feature-selection bash
```

### Внутри контейнера:
```bash
cd /app/vfl-master@73591d69c04
export PYTHONPATH="/app/ucbfl_mock:$PYTHONPATH"
bash run_training_local.sh
```

## ✅ Что было настроено:

1. ✅ Dockerfile обновлен - добавлены grpcio, grpcio-tools, ruamel.yaml
2. ✅ docker-compose.yml обновлен - смонтированы папки vfl-master и ucbfl_mock
3. ✅ network_mode: "host" - для доступа к портам 50000, 50001

## 🔍 Проверка в контейнере

```bash
# Проверка структуры
ls -la /app/vfl-master@73591d69c04/
ls -la /app/ucbfl_mock/

# Проверка зависимостей
python3 -c "import grpc; print('✅ grpc')"
python3 -c "import grpc_tools.protoc; print('✅ grpc_tools')"
python3 -c "import ruamel.yaml; print('✅ ruamel.yaml')"
```

## 🎯 После обучения

Модель будет сохранена в:
- `/app/vfl-master@73591d69c04/example/workdir/active/models/result_model.pkl`
- Доступна на хосте через volume: `./vfl-master@73591d69c04/example/workdir/active/models/result_model.pkl`

## 🔄 Запуск предсказания (после обучения)

```bash
# Внутри контейнера
cd /app/vfl-master@73591d69c04
export PYTHONPATH="/app/ucbfl_mock:$PYTHONPATH"
bash run_prediction_local.sh
```

