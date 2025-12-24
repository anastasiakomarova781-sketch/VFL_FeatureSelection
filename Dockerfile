FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Установка зависимостей ucbfl (без установки самого пакета, используем через PYTHONPATH)
RUN if [ -d "vfl-master@73591d69c04-2/external/UCBFL/python" ]; then \
    cd vfl-master@73591d69c04-2/external/UCBFL/python && \
    pip install --no-cache-dir -r requirements.txt && \
    cd /app; \
    fi

# Генерация proto файлов для VFL
RUN if [ -d "vfl-master@73591d69c04-2" ]; then \
    cd vfl-master@73591d69c04-2 && \
    bash generate_proto.sh && \
    cd /app; \
    fi

# Создание директории для результатов
RUN mkdir -p /app/results

# Установка PYTHONPATH для ucbfl и proto
ENV PYTHONPATH=/app:/app/vfl-master@73591d69c04-2/python:/app/vfl-master@73591d69c04-2/example:/app/vfl-master@73591d69c04-2/external/UCBFL/python

# Команда по умолчанию
CMD ["/bin/bash"]
