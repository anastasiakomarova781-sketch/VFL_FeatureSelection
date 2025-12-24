####################################################
# Dockerfile для VFL Feature Selection Project
# Содержит методы: PSO, FedSDG-FS, VF-PS
####################################################

FROM python:3.11-slim

# Установка системных зависимостей
RUN apt update -y \
    && apt upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        build-essential \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Установка pip и обновление
RUN pip install --no-cache-dir --upgrade pip

# Копирование requirements.txt и установка зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка дополнительных зависимостей
RUN pip install --no-cache-dir \
    xgboost \
    grpcio \
    grpcio-tools \
    ruamel.yaml

# Копирование всего проекта
COPY . .

# Создание рабочей директории для результатов
RUN mkdir -p /app/results

# Установка переменных окружения
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Рабочая директория
WORKDIR /app

# По умолчанию запускаем интерактивную оболочку
CMD ["/bin/bash"]

