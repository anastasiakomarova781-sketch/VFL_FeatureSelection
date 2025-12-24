# Docker Setup для VFL Feature Selection

## 🚀 Быстрый старт

### 1. Сборка образа
```bash
bash docker_build.sh
```
или
```bash
docker-compose build
```

### 2. Запуск контейнера
```bash
bash docker_run.sh
```
или
```bash
docker-compose up -d
```

### 3. Вход в контейнер
```bash
docker-compose exec vfl-feature-selection bash
```

## 📋 Использование

### Запуск FedSDG-FS
```bash
docker-compose exec vfl-feature-selection bash -c "cd /app && python3 fedsdg/run_fedsdg_fs.py"
```

### Запуск PSO
```bash
docker-compose exec vfl-feature-selection bash -c "cd /app && python3 PSO/pso_run.py"
```

### Запуск VF-PS
```bash
docker-compose exec vfl-feature-selection bash -c "cd /app && python3 VF-PS/vf_ps_functions.py"
```

## 📁 Структура в контейнере

- `/app/Data/` - датасеты (read-only)
- `/app/results/` - результаты (read-write)
- `/app/fedsdg/` - метод FedSDG-FS
- `/app/PSO/` - метод PSO
- `/app/VF-PS/` - метод VF-PS

## 🛠 Управление контейнером

### Остановка
```bash
docker-compose stop
```

### Удаление
```bash
docker-compose down
```

### Просмотр логов
```bash
docker-compose logs -f
```

## ✅ Проверка работы

После запуска контейнера проверьте:
```bash
docker-compose exec vfl-feature-selection bash -c "python3 --version && pip list | grep -E '(numpy|pandas|sklearn|phe|xgboost)'"
```
