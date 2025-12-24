# Команды для запуска обучения

## 🎯 Быстрое решение

**Вы находитесь в Docker контейнере!** Нужно выйти и запустить на хосте.

### Шаг 1: Выйти из контейнера
```bash
exit
```

### Шаг 2: Запустить обучение на хосте
```bash
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"
bash run_training_local.sh
```

---

## 🐳 Если хотите запустить в Docker контейнере

### 1. Обновите docker-compose.yml (уже обновлен)

### 2. Пересоздайте контейнер:
```bash
docker-compose down
docker-compose up -d
docker-compose exec vfl-feature-selection bash
```

### 3. Внутри контейнера:
```bash
cd /app/vfl-master@73591d69c04
export PYTHONPATH="/app/ucbfl_mock:$PYTHONPATH"
bash run_training_local.sh
```

---

## ✅ Рекомендуемый способ (с хоста)

```bash
# Выйдите из контейнера
exit

# Запустите обучение
cd /Users/akomarova/Documents/GitHub/VFL_FeatureSelection/vfl-master@73591d69c04
export PYTHONPATH="../ucbfl_mock:$PYTHONPATH"
bash run_training_local.sh
```

---

## 🔍 Как понять, где вы находитесь?

**В контейнере:**
- Приглашение: `root@...:/app#`
- `hostname` показывает ID контейнера
- `pwd` показывает `/app`

**На хосте:**
- Приглашение: `akomarova@IT-MAC-NB223 ... %`
- `pwd` показывает `/Users/akomarova/...`

