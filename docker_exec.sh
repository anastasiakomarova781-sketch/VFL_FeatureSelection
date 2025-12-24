#!/bin/bash
# Скрипт для выполнения команды в Docker контейнере

if [ -z "$1" ]; then
    echo "Использование: $0 <команда>"
    echo "Пример: $0 'python3 fedsdg/run_fedsdg_fs.py'"
    exit 1
fi

docker-compose exec vfl-feature-selection bash -c "$1"
