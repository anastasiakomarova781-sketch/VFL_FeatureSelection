#!/bin/bash
# Скрипт для запуска обучения в Docker контейнере

echo "=== Запуск обучения VFL в Docker контейнере ==="
echo ""

docker-compose exec -T vfl-feature-selection bash -c "
cd /app/vfl-master@73591d69c04
export PYTHONPATH='/app/ucbfl_mock:\$PYTHONPATH'

echo '=== Генерация proto файлов ==='
bash generate_proto.sh

echo ''
echo '=== Запуск пассивного сервера ==='
PYTHONPATH='/app/ucbfl_mock:\$PYTHONPATH' python3 /app/vfl-master@73591d69c04/python/training_server_passive.py --work_dir=/app/vfl-master@73591d69c04/example/workdir/passive > /tmp/passive.log 2>&1 &
PASSIVE_PID=\$!
echo \"Пассивный сервер запущен (PID: \$PASSIVE_PID)\"
sleep 8

echo 'Проверка пассивного сервера:'
tail -3 /tmp/passive.log

echo ''
echo '=== Запуск активного сервера ==='
PYTHONPATH='/app/ucbfl_mock:\$PYTHONPATH' python3 /app/vfl-master@73591d69c04/python/training_server_active.py --work_dir=/app/vfl-master@73591d69c04/example/workdir/active --passive_server_address=localhost:50001 > /tmp/active.log 2>&1 &
ACTIVE_PID=\$!
echo \"Активный сервер запущен (PID: \$ACTIVE_PID)\"
sleep 8

echo 'Проверка активного сервера:'
tail -3 /tmp/active.log

echo ''
echo '=== Запуск клиента обучения ==='
python3 example/training_client.py \\
    --active-dataset=/app/Data/active_dataset_test.csv \\
    --passive-dataset=/app/Data/passive_dataset_test.csv \\
    --match-id-name=id \\
    --label-name=target

echo ''
echo '=== Остановка серверов ==='
kill \$PASSIVE_PID \$ACTIVE_PID 2>/dev/null || true
echo 'Обучение завершено!'
"
