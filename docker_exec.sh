#!/bin/bash

# Скрипт для выполнения команд в запущенном контейнере

if [ $# -eq 0 ]; then
    echo "Использование: $0 <команда>"
    echo ""
    echo "Примеры:"
    echo "  $0 python3 PSO/pso_run.py"
    echo "  $0 python3 fedsdg/run_fedsdg_fs.py"
    echo "  $0 /bin/bash  # Вход в контейнер"
    exit 1
fi

# Проверяем, запущен ли контейнер
if ! docker ps | grep -q "vfl-feature-selection"; then
    echo "Контейнер не запущен. Запускаю..."
    docker start vfl-feature-selection
    sleep 2
fi

# Выполняем команду
docker exec -it vfl-feature-selection "$@"

