#!/bin/bash
# Скрипт для запуска обучения в Docker контейнере

echo "=== Запуск обучения в Docker контейнере ==="
echo ""
echo "Выполните следующие команды:"
echo ""
echo "1. Войдите в контейнер:"
echo "   docker-compose exec vfl-feature-selection bash"
echo ""
echo "2. Внутри контейнера выполните:"
echo "   cd /app/vfl-master@73591d69c04"
echo "   export PYTHONPATH=\"/app/ucbfl_mock:\$PYTHONPATH\""
echo "   bash run_training_local.sh"
echo ""
echo "Или выполните все одной командой:"
echo "   docker-compose exec vfl-feature-selection bash -c 'cd /app/vfl-master@73591d69c04 && export PYTHONPATH=\"/app/ucbfl_mock:\$PYTHONPATH\" && bash run_training_local.sh'"

