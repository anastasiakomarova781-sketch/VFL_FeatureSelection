#!/bin/bash

# Скрипт для локального запуска VFL training клиента
# Требования: установлены grpcio, grpcio-tools, ruamel.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Добавляем mock ucbfl в PYTHONPATH
export PYTHONPATH="$(dirname "$SCRIPT_DIR")/ucbfl_mock:$PYTHONPATH"

echo "=== Генерация proto файлов ==="
if ! python3 -c "import grpc_tools.protoc" 2>/dev/null; then
    echo "Ошибка: grpcio-tools не установлен. Установите его командой:"
    echo "pip3 install grpcio grpcio-tools"
    exit 1
fi

bash generate_proto.sh

echo ""
echo "=== Запуск пассивного сервера (в фоне) ==="
python3 python/training_server_passive.py --work_dir=./example/workdir/passive > passive_server.log 2>&1 &
PASSIVE_PID=$!
echo "Пассивный сервер запущен (PID: $PASSIVE_PID)"

sleep 5

echo ""
echo "=== Ожидание запуска пассивного сервера ==="
sleep 5
# Проверяем, что сервер слушает порт или есть в логах
if [ -f passive_server.log ] && grep -q "listening" passive_server.log 2>/dev/null; then
    echo "✅ Пассивный сервер работает"
elif [ -f passive_server.log ]; then
    echo "⚠️ Проверка логов:"
    tail -3 passive_server.log
    echo "Продолжаем выполнение..."
else
    echo "⚠️ Логи не найдены, но продолжаем..."
fi

echo ""
echo "=== Запуск активного сервера (в фоне) ==="
python3 python/training_server_active.py --work_dir=./example/workdir/active --passive_server_address=localhost:50001 > active_server.log 2>&1 &
ACTIVE_PID=$!
echo "Активный сервер запущен (PID: $ACTIVE_PID)"

sleep 5

echo ""
echo "=== Ожидание запуска активного сервера ==="
sleep 5
# Проверяем, что сервер слушает порт или есть в логах
if [ -f active_server.log ] && grep -q "listening" active_server.log 2>/dev/null; then
    echo "✅ Активный сервер работает"
elif [ -f active_server.log ]; then
    echo "⚠️ Проверка логов:"
    tail -3 active_server.log
    echo "Продолжаем выполнение..."
else
    echo "⚠️ Логи не найдены, но продолжаем..."
fi

echo ""
echo "=== Запуск клиента обучения ==="
python3 example/training_client.py \
    --active-dataset=active_dataset_test.csv \
    --passive-dataset=passive_dataset_test.csv \
    --match-id-name=id \
    --label-name=target

echo ""
echo "=== Остановка серверов ==="
kill $PASSIVE_PID $ACTIVE_PID 2>/dev/null || true
echo "Серверы остановлены"
