#!/bin/bash

# Скрипт для локального запуска VFL prediction клиента
# Требования: установлены grpcio, grpcio-tools, ruamel.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Добавляем mock ucbfl в PYTHONPATH (если существует)
if [ -d "../ucbfl_mock" ]; then
    export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"
fi

echo "=== Генерация proto файлов ==="
if ! python3 -c "import grpc_tools.protoc" 2>/dev/null; then
    echo "Ошибка: grpcio-tools не установлен. Установите его командой:"
    echo "pip3 install grpcio grpcio-tools"
    exit 1
fi

bash generate_proto.sh

echo ""
echo "=== Запуск пассивного сервера предсказания (в фоне) ==="
python3 python/prediction_server_passive.py --work_dir=./example/workdir/passive > passive_prediction_server.log 2>&1 &
PASSIVE_PID=$!
echo "Пассивный сервер предсказания запущен (PID: $PASSIVE_PID)"

sleep 3

echo ""
echo "=== Проверка пассивного сервера ==="
if ps -p $PASSIVE_PID > /dev/null 2>&1; then
    echo "Пассивный сервер работает"
else
    echo "Ошибка: Пассивный сервер не запустился. Логи:"
    cat passive_prediction_server.log
    exit 1
fi

echo ""
echo "=== Запуск активного сервера предсказания (в фоне) ==="
python3 python/prediction_server_active.py --work_dir=./example/workdir/active --passive_server_address=localhost:50051 > active_prediction_server.log 2>&1 &
ACTIVE_PID=$!
echo "Активный сервер предсказания запущен (PID: $ACTIVE_PID)"

sleep 3

echo ""
echo "=== Проверка активного сервера ==="
if ps -p $ACTIVE_PID > /dev/null 2>&1; then
    echo "Активный сервер работает"
else
    echo "Ошибка: Активный сервер не запустился. Логи:"
    cat active_prediction_server.log
    kill $PASSIVE_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "=== Запуск клиента предсказания ==="
python3 example/prediction_client.py \
    --active-dataset=active_dataset_test.csv \
    --passive-dataset=passive_dataset_test.csv \
    --match-id-name=id \
    --model-name=result_model.pkl \
    --scores-name=result_scores.csv

echo ""
echo "=== Остановка серверов ==="
kill $PASSIVE_PID $ACTIVE_PID 2>/dev/null || true
echo "Серверы остановлены"

