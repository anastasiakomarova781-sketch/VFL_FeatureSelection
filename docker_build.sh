#!/bin/bash

# Скрипт для сборки Docker образа VFL Feature Selection

echo "=== Сборка Docker образа VFL Feature Selection ==="

docker build -t vfl-feature-selection:latest .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Образ успешно собран!"
    echo ""
    echo "Для запуска контейнера используйте:"
    echo "  ./docker_run.sh"
    echo ""
    echo "Или:"
    echo "  docker-compose up -d"
else
    echo ""
    echo "❌ Ошибка при сборке образа"
    exit 1
fi

