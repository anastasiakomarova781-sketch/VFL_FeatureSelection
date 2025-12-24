# ОКБ: Вертикальное федеративное обучение. Описание процесса обучения.

# Общая архитектура процесса

Система реализует вертикальное федеративное обучение с использованием алгоритма SecureBoost. В системе участвуют:

- **Активная сторона (Guest)**: Имеет доступ как к признакам (фичам), так и целевым меткам (таргетам). Координирует обучение. Выполняет шифрование/дешифрование.
- **Пассивная сторона (Host)**: Имеет доступ только к признакам (фичам). Выполняет вычисления на гомоморфно зашифрованных данных.
- **RabbitMQ**: Брокер сообщений для обмена данными между участниками.
- **Гомоморфное шифрование Paillier**: Обеспечивает конфиденциальность данных.

## Компоненты используемые для обучения

| Класс                   | Роль                                |
| ----------------------- | ----------------------------------- |
| HeteroSecureBoostGuest  | Главный API класс обучения на Guest |
| HeteroDecisionTreeGuest | Построение деревьев на Guest        |
| HeteroSecureBoostHost   | Главный API класс обучения на Host  |
| HeteroDecisionTreeHost  | Построение деревьев на Host         |
| SBTHistogramBuilder     | Построение гистограмм на Host       |
| SBTSplitter             | Поиск точек разделения              |
| DistributedHistogram    | Хранение зашифрованных статистик    |
| RabbitManager           | Работа с RabbitMQ                   |
| RabbitmqFederation      | Управление защищенной коммуникацией |

## Криптографические компоненты

| Компонент          | Назначение                    |
| ------------------ | ----------------------------- |
| PHETensorEncryptor | Шифрование тензоров           |
| PHETensorDecryptor | Дешифрование тензоров         |
| PHETensorCoder     | Упаковка/распаковка значений  |
| PHETensorCipher    | Управление ключами шифрования |

![диаграма](res/train_diagram_01.svg)
## Общая схема взаимодействия участников
```mermaid
graph TD
    A[Guest] -->|1. Зашифрованные градиенты| B[RabbitMQ]
    B -->|2. Доставка данных| C[Host]
    C -->|3. Агрегированные гистограммы| B
    B -->|4. Доставка гистограмм| A
    A -->|5. Информация о разделении| B
    B -->|6. Доставка разделений| C
```

# Процесс активного обучения на стороне Guest

## Инициализация

1. Генерация ключей Paillier (открытый `pk` и секретный `sk`).
2. Биннинг данных и инициализация градиентов.

## Шифрование данных

```python
### Шифрование градиентов и гессианов
def _g_h_process(grad_and_hess):
    # 1. Упаковка g/h
    pack_tensor = torch.Tensor([g + g_offset, h])
    
    # 2. Преобразование в фиксированную точку
    pack_vec = coder.pack_floats(
        pack_tensor, 
        shift_bit, 
        pack_num=2, 
        precision=FIX_POINT_PRECISION
    )
    
    # 3. Шифрование Paillier
    en = pk.encrypt_encoded(pack_vec, obfuscate=True)
    return en
```

### Дешифровка данных (гистограмм)
```python
def split(ctx, statistic_histogram):
    # 1. Дешифровка с использованием sk
    decrypted_hist = statistic_histogram.decrypt(
        sk_map={"gh": self._sk},
        coder_map={"gh": (self._coder, torch.float32)}
    )
    
    # 2. Поиск оптимального разделения
    for feature, bin in decrypted_hist:
        gain = self._compute_gain(hist[feature][bin])
        if gain > best_gain and gain > min_impurity_split:
            best_split = (feature, bin)
    
    return best_split
```
## Основной цикл обучения

1. **Отправка данных  на сторону Host**:

- `en_gh`: Зашифрованные градиенты/гессианы
- `en_kit`: Открытый ключ и evaluator
- `pack_info`: Параметры упаковки

2. **Получение гистограмм**:

- Получает `DistHist` от Host (агрегированные зашифрованные гистограммы)

3. **Дешифровка и поиск разделений**:

- Дешифровка с использованием секретного ключа `sk`

- Поиск лучшего разделения по критерию gain

4. **Синхронизация**:

- Отправка маскированной структуры дерева (`sync_nodes`)

- Обновление позиций образцов (`updated_data`)

## Финализация

- Сохранение модели: `save_model(output_model_path)`

## Схема активного обучения на стороне Guest

![диаграма](res/train_diagram_02.svg)
```mermaid
sequenceDiagram
    participant GuestApp
    participant SBoostG as SecureBoostGuest
    participant DTreeG as DecisionTreeGuest
    participant Encryptor
    participant RabbitMQ
    
    GuestApp->>SBoostG: fit(ctx, data_frame)
    SBoostG->>DTreeG: booster_fit()
    DTreeG->>Encryptor: Инициализация Paillier
    Encryptor-->>DTreeG: pk, sk
    DTreeG->>DTreeG: _g_h_process() # Шифрование
    DTreeG->>RabbitMQ: en_gh, en_kit, pack_info
    
    loop Для каждого уровня дерева
        RabbitMQ->>DTreeG: DistHist
        DTreeG->>DTreeG: decrypt() # Дешифровка
        DTreeG->>DTreeG: Поиск split_info
        DTreeG->>RabbitMQ: sync_nodes, updated_data
    end
    
    DTreeG->>SBoostG: Обученное дерево
    SBoostG->>GuestApp: get_model()
```

# Процесс пассивного обучения на стороне Host

## Инициализация

1. Получение зашифрованных градиентов (`en_gh`) и криптоинструментов (`en_kit`).

2. Инициализация гистограмм.

## Гомоморфная агрегация
Host выполняет **только операции сложения** на зашифрованных данных:

```python
class HistogramValuesContainer:
    def i_update(self, targets, positions):
        for name, value in targets.items():
            # Делегирование операции соответствующему типу значений
            self._data[name].i_update(value, positions)
    
    def iadd(self, other: "HistogramValuesContainer"):
        for name, values in other._data.items():
            if name in self._data:
                self._data[name].iadd(values)
```
```python
class HistogramEncryptedValues:
    def i_update(self, value, positions):
        for pos in positions:
            # Гомоморфное сложение: E(a) + E(b)
            self.data[pos] = self.evaluator.add(self.pk, self.data[pos], value)
    
    def iadd(self, other):
        for i in range(len(self.data)):
            # Гомоморфное сложение массивов
            self.data[i] = self.evaluator.add(self.pk, self.data[i], other.data[i])
```
## Основной цикл обучения

1. **Построение гистограмм**:

- Агрегация зашифрованных значений по бинам и узлам дерева.

- Возврат сжатых гистограмм (`DistHist`) на Guest.

2. **Получение информации о разделении**:

- Получает `sync_nodes` от Guest (маскированная структура дерева).

3. **Обновление позиций образцов**:

- Локальное обновление позиций на основе правил разделения.

- Отправка обновленных позиций (`updated_data`).

## Особенности работы

- **Безопасность**: Host никогда не видит расшифрованные градиенты.

- **Эффективность**: Используется пакетная обработка и сжатие данных.

## Схема пассивного обучения на стороне Host
![диаграма](res/train_diagram_03.svg)
```mermaid
sequenceDiagram
    participant RabbitMQ
    participant DTreeH as DecisionTreeHost
    participant HistBuilder
    participant Histogram
    participant ValuesContainer
    
    RabbitMQ->>DTreeH: en_gh, en_kit
    DTreeH->>HistBuilder: compute_hist()
    
    loop Для каждого образца данных
        HistBuilder->>Histogram: i_update(feature, node, encrypted_value)
        Histogram->>ValuesContainer: i_update(encrypted_value, position)
        ValuesContainer->>ValuesContainer: Гомоморфное сложение
    end
    
    HistBuilder->>DTreeH: statistic_histogram
    DTreeH->>RabbitMQ: DistHist
    RabbitMQ->>DTreeH: sync_nodes
    DTreeH->>DTreeH: Обновление позиций образцов
```

# Взаимодействие участников через RabbitMQ

## Типы сообщений

| Тип          | Направление      | Описание                           | Шифрование    |
|--------------|------------------|------------------------------------|----------------|
| **en_gh**    | Guest → Host   | Зашифрованные градиенты            | Paillier       |
| **en_kit**   | Guest → Host   | pk, evaluator                      | Открытый ключ  |
| **pack_info**| Guest → Host   | Параметры упаковки                 | Открытый текст |
| **DistHist** | Host → Guest   | Агрегированные гистограммы         | Paillier       |
| **sync_nodes**| Guest → Host | Маскированная структура дерева     | Открытый текст |
| **updated_data** | Обе стороны  | Обновленные позиции образцов      | Открытый текст |


## Управление очередями

```python
class RabbitManager:
    def create_queue(vhost, queue_name): ...
    def bind_exchange_to_queue(vhost, exchange, queue): ...
    def federate_queue(upstream, vhost, send_q, receive_q): ...
    def put(key, value): ...  # Отправка сообщения
    def get(key, timeout=None): ...  # Получение сообщения
```

## Детали шифрования и безопасности

### Схема шифрования Paillier

- **Гомоморфные свойства**: Поддерживает сложение зашифрованных значений.
- **Упаковка данных**: Несколько значений упаковываются в одно число для эффективности.
- **Фиксированная точка**: Преобразование чисел с плавающей точкой в целые для шифрования.

### Параметры безопасности

- **Длина ключа**: 1024 бита (по умолчанию).
- **Обфускация**: Добавление случайного шума при шифровании.
- **Маскировка метаданных**: Скрытие чувствительной информации о дереве.

# Диаграммы

## Общая схема обучения

![диаграма](res/train_diagram_04.svg)
```mermaid

sequenceDiagram

participant GuestApp
participant Guest
participant RabbitMQ
participant Host
participant HostApp

GuestApp->>Guest: fit(data)
Guest->>Guest: Генерация ключей
Guest->>RabbitMQ: en_gh, en_kit, pack_info
RabbitMQ->>Host: Доставка
Host->>Host: Агрегация гистограмм
Host->>RabbitMQ: DistHist
RabbitMQ->>Guest: Доставка
Guest->>Guest: Дешифровка

loop Для каждого уровня дерева
Guest->>Guest: Поиск разделений
Guest->>RabbitMQ: sync_nodes, updated_data
RabbitMQ->>Host: Доставка
Host->>Host: Обновление позиций
Host->>RabbitMQ: updated_data
RabbitMQ->>Guest: Доставка
end

Guest->>Guest: Финализация дерева
Guest->>GuestApp: Модель
Host->>HostApp: Модель

```

## Детализированная схема обучения
![диаграма](res/train_diagram_05.svg)
```mermaid
sequenceDiagram
    participant GA as GuestApp
    participant SG as SecureBoostGuest
    participant TG as TreeGuest
    participant SPL as SBTSplitter
    participant RMQ as RabbitMQ
    participant TH as TreeHost
    participant SH as SecureBoostHost
    participant DH as DistributedHistogram
    participant HGB as HistogramBuilder
    participant HB as SBTHistogramBuilder

    GA->>SG: fit(ctx, data_frame)
    note right of SG: 1. Биннинг данных<br/>2. Инициализация градиентов<br/>3. Создание деревьев

    rect rgb(220, 240, 255)
        SG->>RMQ: [инициализация обучения]
        RMQ->>SH: [начало обучения]
        note right of SH: 1. Получение данных<br/>2. Инициализация деревьев<br/>3. Подготовка к обучению
        SH->>TH: booster_fit(bin_data, binning_dict)
        note right of TH: Ожидание зашифрованных данных<br/>и крипто-инструментов
    end

    SG->>TG: booster_fit(bin_data, grad_and_hess, binning_dict)
    note right of TG: Запуск построения дерева решения

    rect rgb(240, 255, 220)
        TG->>TG: _g_h_process(grad_and_hess)
        note left of TG: Шифрование градиентов:<br/>- Упаковка g/h в вектор<br/>- Paillier-шифрование<br/>- Настройка смещения (g_offset)<br/>- Фиксированная точка (shift_bit)
        TG->>RMQ: PUT en_gh (зашифр. градиенты/гессианы)
        TG->>RMQ: PUT en_kit (pk, evaluator)
        TG->>RMQ: PUT pack_info (total_pack_num, split_point_shift_bit)
        RMQ->>TH: [en_gh, en_kit, pack_info]
        note right of TH: Получение крипто-инструментов<br/>и зашифрованных производных
    end

    loop Для каждого уровня глубины (depth = 0 → max_depth)
        rect rgb(255, 230, 230)
            TH->>HB: compute_hist(ctx, nodes, bin_train_data, en_gh)
            note right of HB: Создание схемы шифрования<br/>на основе en_kit
            HB->>HGB: statistic(data)
            note right of HGB: Гомоморфная агрегация:<br/>- mapReducePartitions<br/>- iadd зашифрованных значений
            HGB->>DH: Создание объекта
            DH-->>TH: Гистограмма
            note right of DH: Зашифрованные суммы:<br/>- g (градиенты)<br/>- h (гессианы)<br/>- cnt (количество)
            TH->>RMQ: PUT DistHist (зашифр. гистограммы)
            RMQ->>TG: [DistHist]
            note left of TG: Получение агрегированных статистик<br/>по признакам Host
        end

        rect rgb(230, 255, 230)
            TG->>SPL: split(ctx, DistHist, ...)
            note left of SPL: 1. Дешифровка гистограмм (sk)<br/>2. _compute_gains() на зашифр. данных<br/>3. Выбор max gain<br/>4. Проверка min_impurity_split
            SPL->>DH: decrypt(sk_map, coder_map)
            note left of DH: Расшифровка только на Guest<br/>с приватным ключом
            DH-->>SPL: Расшифрованные гистограммы
            SPL-->>TG: Лучшие split_info
        end

        rect rgb(230, 230, 255)
            TG->>TG: _sync_nodes(cur_nodes, next_nodes)
            note left of TG: Маскировка чувствительной информации:<br/>- split_id → fid/bid<br/>- sitename<br/>- gain
            TG->>RMQ: PUT sync_nodes
            RMQ->>TH: [sync_nodes]
            note right of TH: Получение структуры дерева<br/>без деталей сплитов
            TH->>SH: [обновление состояния дерева]
            note right of SH: Обновление внутреннего состояния
        end

        rect rgb(255, 255, 200)
            TG->>TG: _update_sample_pos(..., local_update=False)
            note left of TG: Частичное обновление позиций<br/>на локальных данных
            TG->>RMQ: PUT updated_data
            RMQ->>TH: [updated_data]
            TH->>TH: _update_sample_pos(..., node_map)
            note right of TH: Обновление позиций<br/>на основе локальных правил
            TH->>RMQ: PUT updated_data
            RMQ->>TG: [updated_data]
            TG->>TG: _merge_sample_positions()
            note left of TG: Объединение позиций от всех участников
            TG->>RMQ: PUT new_sample_pos
            RMQ->>TH: [new_sample_pos]
            note right of TH: Финальные позиции образцов<br/>для следующего уровня
            TH->>SH: [прогресс обучения]
            note right of SH: Обновление метрик<br/>и состояния обучения
        end
    end

    rect rgb(200, 255, 200)
        TH->>SH: [финальное состояние дерева]
        note right of SH: 1. Обновление feature_importance<br/>2. Финализация модели
        TG->>SG: [обученное дерево]
        note left of TG: - Веса листьев<br/>- Информация о разделениях<br/>- Важность признаков
        SG->>GA: get_model()
        GA->>GA: save_model(output_model_path)
        note left of GA: Сериализация через pickle
        SH->>SH: save_model()
        note right of SH: Сохранение модели на Host
    end
```

## Детализированная схема обучения совместимая с https://www.sequencediagram.org/
[train_diagramm](res/train_diagramm.txt)

# Заключение

Система обеспечивает конфиденциальность данных за счет использования гомоморфного шифрования и безопасного обмена сообщениями. Guest координирует обучение и выполняет криптографические операции, в то время как Host участвует в вычислениях, не имея доступа к сырым данным.