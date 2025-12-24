#!/bin/bash

# Скрипт для запуска Docker контейнера VFL Feature Selection

echo "=== Запуск Docker контейнера VFL Feature Selection ==="

# Создаем папку для результатов, если её нет
mkdir -p results

# Проверяем, существует ли образ
if ! docker images | grep -q "vfl-feature-selection"; then
    echo "Образ не найден. Запускаю сборку..."
    ./docker_build.sh
fi

# Останавливаем существующий контейнер, если есть
docker stop vfl-feature-selection 2>/dev/null
docker rm vfl-feature-selection 2>/dev/null

# Запускаем контейнер
echo ""
echo "Запуск контейнера..."
docker run -it \
    --name vfl-feature-selection \
    -v "$(pwd)/Data:/app/Data:ro" \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/PSO:/app/PSO" \
    -v "$(pwd)/fedsdg:/app/fedsdg" \
    -v "$(pwd)/VF-PS:/app/VF-PS" \
    -w /app \
    vfl-feature-selection:latest \
    /bin/bash

echo ""
echo "Контейнер остановлен."

