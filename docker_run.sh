#!/bin/bash
# Скрипт для запуска Docker контейнера

echo "=== Запуск Docker контейнера ==="
docker-compose up -d

echo ""
echo "=== Вход в контейнер ==="
docker-compose exec vfl-feature-selection /bin/bash
