# Решение проблемы SSL сертификатов для Docker

## Проблема
Ошибка `tls: failed to verify certificate: x509: certificate signed by unknown authority` возникает при попытке Docker подключиться к приватному репозиторию `nexus.ucb.infra`.

## ✅ Решение (уже применено)

Мы **уже решили проблему**, убрав зависимость от приватного репозитория:
- Убрали установку ucbfl через `pip install -e .` (которая требовала Rust компоненты)
- Используем ucbfl через PYTHONPATH вместо установки пакета
- Docker образ собирается успешно без доступа к приватному репозиторию

## Если все же нужно добавить SSL сертификат

### Вариант 1: Добавить сертификат в Docker образ (рекомендуется)

Добавьте в Dockerfile перед установкой зависимостей:

```dockerfile
# Добавление SSL сертификата для приватного репозитория
RUN mkdir -p /usr/local/share/ca-certificates/ && \
    echo "-----BEGIN CERTIFICATE-----" > /usr/local/share/ca-certificates/nexus.crt && \
    echo "ВАШ_СЕРТИФИКАТ_ЗДЕСЬ" >> /usr/local/share/ca-certificates/nexus.crt && \
    echo "-----END CERTIFICATE-----" >> /usr/local/share/ca-certificates/nexus.crt && \
    update-ca-certificates
```

### Вариант 2: Добавить сертификат в систему хоста (требует sudo)

```bash
# Получить сертификат
hostname="nexus.ucb.infra"
port=443
echo -n | openssl s_client -showcerts -connect $hostname:$port -servername $hostname 2>/dev/null | \
    sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' | \
    sudo tee -a /etc/ssl/certs/ca-certificates.crt

# Перезапустить Docker daemon (на macOS через Docker Desktop)
```

### Вариант 3: Отключить проверку SSL (только для разработки, НЕ рекомендуется)

В Dockerfile добавьте:
```dockerfile
ENV PYTHONHTTPSVERIFY=0
ENV CURL_CA_BUNDLE=""
```

## Текущее решение (работает без приватного репозитория)

Наш Dockerfile уже настроен правильно:
- Устанавливает только зависимости из requirements.txt
- Использует ucbfl через PYTHONPATH
- Не требует доступа к приватному репозиторию

**Вывод:** Дополнительные действия не требуются, все уже работает!

