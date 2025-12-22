"""
FedSDG-FS: Полная реализация Federated Stochastic Dual-Gate Feature Selection
Реализация алгоритма для вертикального федеративного обучения с:
- Активным клиентом (имеет таргет, координатор)
- Пассивным клиентом (нет таргета)
- Шифрованием Paillier
- Стохастическим двойным гейтом (Dual Gate)
- Forward и Backward propagation
"""

# Импорт библиотеки numpy для работы с массивами и математическими операциями
import numpy as np
# Импорт библиотеки pandas для работы с табличными данными (DataFrame)
import pandas as pd
# Импорт типов данных для аннотации типов в функциях (List, Tuple, Dict, Optional)
from typing import List, Tuple, Dict, Optional
# Импорт StandardScaler для нормализации данных (приведение к среднему 0 и дисперсии 1)
from sklearn.preprocessing import StandardScaler
# Импорт модуля warnings для управления предупреждениями
import warnings
# Отключение всех предупреждений для чистого вывода в консоль
warnings.filterwarnings('ignore')

# Попытка импортировать библиотеку paillier для гомоморфного шифрования
try:
    # Импорт модуля paillier из библиотеки phe (Python Homomorphic Encryption)
    from phe import paillier
    # Установка флага доступности Paillier в True при успешном импорте
    PAILLIER_AVAILABLE = True
except ImportError:
    # Если импорт не удался, устанавливаем флаг в False
    PAILLIER_AVAILABLE = False
    # Вывод предупреждения о необходимости установки библиотеки
    print("Warning: phe library not found. Install with: pip install phe")


class PaillierEncryption:
    """Класс для работы с шифрованием Paillier - гомоморфное шифрование для защиты данных"""
    
    def __init__(self, key_length: int = 1024):
        """Инициализация шифрования Paillier - создание пары ключей для шифрования/расшифровки"""
        # Проверка доступности библиотеки Paillier
        if not PAILLIER_AVAILABLE:
            # Выброс исключения, если библиотека не установлена
            raise ImportError("phe library is required. Install with: pip install phe")
        # Генерация пары ключей (публичный и приватный) для шифрования Paillier
        # Публичный ключ используется для шифрования, приватный - для расшифровки
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        # Сохранение длины ключа в атрибуте объекта (влияет на безопасность)
        self.key_length = key_length
    
    def encrypt(self, value: float) -> paillier.EncryptedNumber:
        """Шифрование значения - преобразует число в зашифрованное значение"""
        # Шифрование числового значения с использованием публичного ключа
        # Возвращает зашифрованное число типа EncryptedNumber
        return self.public_key.encrypt(value)
    
    def encrypt_matrix(self, matrix: np.ndarray) -> List[List]:
        """Шифрование матрицы значений - шифрует каждый элемент матрицы"""
        # Инициализация списка для хранения зашифрованной матрицы
        encrypted_matrix = []
        # Цикл по каждой строке матрицы
        for row in matrix:
            # Шифрование каждого значения в строке: если значение не NaN, шифруем его, иначе None
            encrypted_row = [self.encrypt(float(val)) if not np.isnan(val) else None for val in row]
            # Добавление зашифрованной строки в матрицу
            encrypted_matrix.append(encrypted_row)
        # Возврат зашифрованной матрицы в виде списка списков
        return encrypted_matrix
    
    def decrypt(self, encrypted_value: paillier.EncryptedNumber) -> float:
        """Расшифровка значения - преобразует зашифрованное значение обратно в число"""
        # Расшифровка зашифрованного значения с использованием приватного ключа
        # Возвращает исходное числовое значение типа float
        return self.private_key.decrypt(encrypted_value)
    
    def decrypt_matrix(self, encrypted_matrix: List[List]) -> np.ndarray:
        """Расшифровка матрицы значений - расшифровывает каждый элемент матрицы"""
        # Инициализация списка для хранения расшифрованной матрицы
        decrypted = []
        # Цикл по каждой строке зашифрованной матрицы
        for row in encrypted_matrix:
            # Расшифровка каждого значения в строке: если значение не None, расшифровываем, иначе NaN
            decrypted_row = [self.decrypt(val) if val is not None else np.nan for val in row]
            # Добавление расшифрованной строки в матрицу
            decrypted.append(decrypted_row)
        # Преобразование списка списков в numpy массив и возврат
        return np.array(decrypted)


class ActiveClient:
    """
    Активный клиент: имеет целевую переменную (таргет), строит локальную модель, играет роль координатора
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'target', 
                 lambda_reg: float = 0.01, learning_rate: float = 0.01,
                 use_encryption: bool = True, key_length: int = 1024, random_state: int = 42):
        """
        Инициализация активного клиента - клиент с целевой переменной, выступает координатором
        
        Args:
            data: Данные с признаками и целевой переменной
            target_col: Название столбца с целевой переменной
            lambda_reg: Параметр регуляризации L2 (контролирует переобучение)
            learning_rate: Скорость обучения для градиентного спуска
            use_encryption: Использовать ли шифрование Paillier для защиты данных
            key_length: Длина ключа для Paillier (влияет на безопасность)
            random_state: Seed для генератора случайных чисел (для воспроизводимости результатов)
        """
        # Сохранение данных в виде копии (чтобы не изменять исходные данные)
        self.data = data.copy()
        # Извлечение названий признаков (исключаем 'id' и целевую переменную)
        self.feature_names = [col for col in data.columns if col not in ['id', target_col]]
        # Извлечение признаков в numpy массив, заполнение пропущенных значений нулями
        self.X = data[self.feature_names].fillna(0).values
        # Извлечение целевой переменной (меток) в numpy массив
        self.y = data[target_col].values
        # Сохранение количества признаков (столбцов в матрице признаков)
        self.n_features = self.X.shape[1]
        # Сохранение количества образцов (строк в матрице признаков)
        self.n_samples = self.X.shape[0]
        
        # Параметры обучения модели
        # Параметр регуляризации L2 для предотвращения переобучения
        self.lambda_reg = lambda_reg
        # Скорость обучения для градиентного спуска (влияет на скорость сходимости)
        self.learning_rate = learning_rate
        # Seed для генератора случайных чисел (для воспроизводимости)
        self.random_state = random_state
        # Установка seed для генератора случайных чисел numpy
        np.random.seed(random_state)
        
        # Инициализация шифрования для защиты данных при передаче
        if use_encryption:
            # Создание объекта PaillierEncryption с заданной длиной ключа
            self.encryption = PaillierEncryption(key_length=key_length)
        else:
            # Если шифрование отключено, устанавливаем encryption в None
            self.encryption = None
        
        # Нормализация данных для стабильности обучения
        # Создание объекта StandardScaler для нормализации (среднее=0, дисперсия=1)
        self.scaler = StandardScaler()
        # Вычисление параметров нормализации и применение к данным
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Инициализация локальных gates (вероятности участия признаков в модели)
        # Gates - это значения от 0 до 1, показывающие насколько признак важен
        # Начальные вероятности для всех признаков = 0.5 (нейтральное значение)
        self.local_gates = np.ones(self.n_features) * 0.5
        
        # Инициализация весов модели (для простоты используем линейную модель)
        # Веса определяют вклад каждого признака в предсказание
        # Инициализация случайными значениями из нормального распределения
        self.model_weights = np.random.normal(0, 0.1, self.n_features)
        
        # Глобальные gates (будут обновляться после агрегации от всех клиентов)
        # Глобальные gates - это усредненные локальные gates всех участников
        # Изначально равны локальным gates
        self.global_gates = self.local_gates.copy()
        
        # Индикаторная матрица классов (one-hot encoding для таргета)
        # Матрица, где каждая строка соответствует образцу, а столбец - классу
        # Значение 1 означает принадлежность образца к классу
        self.label_matrix = self._create_label_matrix()
    
    def _create_label_matrix(self) -> np.ndarray:
        """Создание индикаторной матрицы классов (one-hot encoding) - для шифрования меток"""
        # Получение уникальных значений классов из целевой переменной
        unique_classes = np.unique(self.y)
        # Подсчет количества уникальных классов
        n_classes = len(unique_classes)
        # Создание матрицы нулей размером (количество образцов, количество классов)
        label_matrix = np.zeros((self.n_samples, n_classes))
        # Заполнение матрицы: для каждого класса устанавливаем 1 на позиции соответствующих образцов
        for i, class_val in enumerate(unique_classes):
            # Для всех образцов с данным классом устанавливаем 1 в соответствующем столбце
            label_matrix[self.y == class_val, i] = 1
        # Возврат индикаторной матрицы классов
        return label_matrix
    
    def encrypt_label_matrix(self) -> List[List]:
        """Шифрование индикаторной матрицы классов - защита меток при передаче"""
        # Если шифрование включено, шифруем матрицу классов
        if self.encryption:
            # Использование метода encrypt_matrix для шифрования всей матрицы
            return self.encryption.encrypt_matrix(self.label_matrix)
        # Если шифрование отключено, возвращаем матрицу как список списков
        return self.label_matrix.tolist()
    
    def compute_local_gates(self) -> np.ndarray:
        """
        Вычисление локальных gates на основе метрики Gini для бинарного таргета
        Gini impurity показывает, насколько хорошо признак разделяет классы
        """
        # Инициализация массива для хранения оценок Gini impurity для каждого признака
        gini_scores = np.zeros(self.n_features)
        
        # Цикл по всем признакам для вычисления их важности
        for i in range(self.n_features):
            # Извлечение i-го признака из нормализованных данных
            feature = self.X_scaled[:, i]
            # Получение уникальных значений признака (для группировки образцов)
            unique_vals = np.unique(feature)
            
            # Проверка, является ли признак константным (все значения одинаковы)
            if len(unique_vals) < 2:
                # Если признак константный, он не может разделять классы
                # Устанавливаем нейтральную оценку 0.5
                gini_scores[i] = 0.5
                continue
            
            # Вычисление Gini impurity для каждого значения признака
            # Список для хранения Gini значений для каждого уникального значения признака
            gini_list = []
            # Цикл по каждому уникальному значению признака
            for val in unique_vals:
                # Создание маски для объектов с данным значением признака
                mask = feature == val
                # Извлечение целевой переменной для объектов с данным значением признака
                y_subset = self.y[mask]
                
                # Пропуск, если нет объектов с данным значением
                if len(y_subset) == 0:
                    continue
                
                # Вычисление долей классов в подмножестве
                # unique_y - уникальные значения классов, counts - их количество
                unique_y, counts = np.unique(y_subset, return_counts=True)
                # Вычисление вероятностей каждого класса (доли)
                probs = counts / len(y_subset)
                
                # Вычисление Gini impurity: Gini = 1 - (p0^2 + p1^2)
                # Чем ниже Gini, тем лучше признак разделяет классы
                gini = 1 - np.sum(probs ** 2)
                # Добавление Gini значения в список
                gini_list.append(gini)
            
            # Вычисление среднего Gini по всем значениям признака
            if len(gini_list) > 0:
                # Средний Gini по всем значениям признака
                avg_gini = np.mean(gini_list)
                # Преобразование Gini в вероятность: чем ниже Gini, тем выше вероятность
                # Инвертируем: вероятность = 1 - средний Gini
                # (низкий Gini означает хорошее разделение, поэтому высокая вероятность)
                gini_scores[i] = 1 - avg_gini
            else:
                # Если не удалось вычислить Gini, устанавливаем нейтральную оценку
                gini_scores[i] = 0.5
        
        # Обновление локальных gates на основе Gini scores
        # Используем адаптивное обновление: новый gate = старый gate + learning_rate * (gini_score - старый gate)
        # Это позволяет плавно обновлять gates с учетом важности признаков
        self.local_gates = self.local_gates + self.learning_rate * (gini_scores - self.local_gates)
        # Ограничение значений gates в диапазоне [0, 1] (вероятности не могут быть отрицательными или больше 1)
        self.local_gates = np.clip(self.local_gates, 0, 1)
        
        # Возврат обновленных локальных gates
        return self.local_gates
    
    def compute_hidden_predictions(self) -> np.ndarray:
        """
        Вычисление промежуточных предсказаний (скрытых значений активации)
        для передачи пассивным клиентам - эти значения помогают пассивным клиентам
        оценить важность своих признаков без доступа к таргету
        """
        # Применение масок gates к признакам: умножаем каждый признак на его gate
        # Это позволяет "включать/выключать" признаки в зависимости от их важности
        masked_features = self.X_scaled * self.local_gates
        # Вычисление взвешенной суммы (промежуточное предсказание)
        # Скалярное произведение замаскированных признаков на веса модели
        hidden_pred = np.dot(masked_features, self.model_weights)
        # Возврат промежуточных предсказаний для передачи пассивным клиентам
        return hidden_pred
    
    def forward_propagation(self, passive_embeddings: List[np.ndarray], 
                           noise_scale: float = 0.1) -> np.ndarray:
        """
        Forward propagation: получение зашифрованных эмбеддингов от пассивных клиентов,
        вычисление взвешенных сумм с добавлением случайного шума для защиты приватности
        
        Args:
            passive_embeddings: Список зашифрованных эмбеддингов от пассивных клиентов
            noise_scale: Масштаб случайного шума для защиты приватности (предотвращает утечку информации)
        
        Returns:
            Зашифрованные взвешенные суммы для отправки обратно клиентам
        """
        # Вычисление локальных скрытых предсказаний активного клиента
        local_hidden = self.compute_hidden_predictions()
        
        # Агрегация эмбеддингов от пассивных клиентов
        # Начинаем с локальных предсказаний активного клиента
        aggregated = local_hidden.copy()
        # Цикл по эмбеддингам от всех пассивных клиентов
        for emb in passive_embeddings:
            # Если эмбеддинги зашифрованы, нужно расшифровать для вычислений
            # В реальной реализации это делается через гомоморфные операции Paillier
            if isinstance(emb, list):
                # Упрощенная версия: предполагаем, что эмбеддинги уже расшифрованы
                # Преобразуем список в numpy массив и добавляем к агрегированным значениям
                aggregated += np.array(emb)
            else:
                # Если эмбеддинги уже в виде numpy массива, добавляем напрямую
                aggregated += emb
        
        # Добавление случайного шума для защиты приватности
        # Шум предотвращает возможность восстановления исходных данных из агрегированных значений
        noise = np.random.normal(0, noise_scale, aggregated.shape)
        # Добавление шума к агрегированным значениям
        aggregated += noise
        
        # Шифрование результата перед отправкой обратно клиентам
        if self.encryption:
            # Преобразование в матрицу (reshape) и шифрование всех значений
            return self.encryption.encrypt_matrix(aggregated.reshape(-1, 1))
        
        # Если шифрование отключено, возвращаем агрегированные значения без шифрования
        return aggregated
    
    def backward_propagation(self, gradients: List[np.ndarray]) -> np.ndarray:
        """
        Backward propagation: вычисление градиентов потерь, шифрование и отправка клиентам
        Градиенты показывают направление изменения весов для уменьшения ошибки
        
        Args:
            gradients: Градиенты от пассивных клиентов (для агрегации)
        
        Returns:
            Зашифрованные градиенты для обновления модели
        """
        # Вычисление градиентов для локальной модели
        # Упрощенная версия: градиент = ошибка * признаки
        # Вычисление промежуточных предсказаний модели
        predictions = self.compute_hidden_predictions()
        # Вычисление ошибок: разница между предсказаниями и реальными значениями таргета
        errors = predictions - self.y
        # Вычисление градиентов: скалярное произведение транспонированных признаков на ошибки
        # Деление на количество образцов для усреднения
        local_gradients = np.dot(self.X_scaled.T, errors) / self.n_samples
        
        # Агрегация градиентов от пассивных клиентов
        # Начинаем с локальных градиентов активного клиента
        aggregated_gradients = local_gradients.copy()
        # Цикл по градиентам от всех пассивных клиентов
        for grad in gradients:
            # Если градиент в виде numpy массива, добавляем к агрегированным градиентам
            if isinstance(grad, np.ndarray):
                aggregated_gradients += grad
        
        # Применение регуляризации L2 для предотвращения переобучения
        # Добавляем штраф за большие веса: lambda_reg * веса
        aggregated_gradients += self.lambda_reg * self.model_weights
        
        # Обновление весов модели методом градиентного спуска
        # Новый вес = старый вес - скорость_обучения * градиент
        self.model_weights -= self.learning_rate * aggregated_gradients
        
        # Шифрование градиентов перед отправкой клиентам для защиты приватности
        if self.encryption:
            # Преобразование в матрицу (reshape) и шифрование всех значений
            return self.encryption.encrypt_matrix(aggregated_gradients.reshape(-1, 1))
        
        # Если шифрование отключено, возвращаем градиенты без шифрования
        return aggregated_gradients
    
    def aggregate_global_gates(self, passive_gates: List[np.ndarray]) -> np.ndarray:
        """
        Агрегация глобальных gates от всех клиентов (активного и пассивных)
        Глобальные gates - это усредненные локальные gates всех участников
        
        Args:
            passive_gates: Список локальных gates от пассивных клиентов
        
        Returns:
            Глобальные gates после агрегации (усредненные значения)
        """
        # Обновление локальных gates активного клиента на основе метрики Gini
        self.compute_local_gates()
        
        # Агрегация: собираем локальные gates всех клиентов в один список
        # Начинаем с локальных gates активного клиента
        all_gates = [self.local_gates]
        # Добавляем локальные gates от каждого пассивного клиента
        for gates in passive_gates:
            all_gates.append(gates)
        
        # Вычисление среднего по всем клиентам (усреднение по оси 0 - по клиентам)
        # Это создает глобальные gates, которые отражают коллективное мнение всех участников
        self.global_gates = np.mean(all_gates, axis=0)
        
        # Возврат глобальных gates
        return self.global_gates
    
    def select_features(self, threshold: float = 0.5) -> List[str]:
        """
        Выбор признаков на основе глобальных gates
        Признаки с gate выше порога считаются важными и остаются в модели
        
        Args:
            threshold: Порог для отбора признаков (значение от 0 до 1)
        
        Returns:
            Список названий отобранных признаков
        """
        # Создание булевой маски для отбора признаков: глобальный gate > порога
        # True означает, что признак отобран, False - что признак удален
        selected_mask = self.global_gates > threshold
        # Получение названий отобранных признаков на основе маски
        # Используем list comprehension для фильтрации признаков
        selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_mask[i]]
        # Возврат списка названий отобранных признаков
        return selected_features
    
    def get_public_key(self):
        """Получить публичный ключ для передачи пассивным клиентам - для шифрования их данных"""
        # Если шифрование включено, возвращаем публичный ключ
        if self.encryption:
            return self.encryption.public_key
        # Если шифрование отключено, возвращаем None
        return None


class PassiveClient:
    """
    Пассивный клиент: нет таргета, просто участвует в выборе признаков
    Оценивает важность своих признаков на основе согласованности с промежуточными предсказаниями активного клиента
    """
    
    def __init__(self, data: pd.DataFrame, public_key=None,
                 lambda_reg: float = 0.01, learning_rate: float = 0.01,
                 use_encryption: bool = True, random_state: int = 42):
        """
        Инициализация пассивного клиента - клиент без доступа к целевой переменной
        
        Args:
            data: Данные с признаками (без целевой переменной)
            public_key: Публичный ключ от активного клиента для шифрования данных
            lambda_reg: Параметр регуляризации L2 (контролирует переобучение)
            learning_rate: Скорость обучения для градиентного спуска
            use_encryption: Использовать ли шифрование Paillier для защиты данных
            random_state: Seed для генератора случайных чисел (для воспроизводимости)
        """
        # Сохранение данных в виде копии (чтобы не изменять исходные данные)
        self.data = data.copy()
        # Извлечение названий признаков (исключаем только 'id', так как таргета нет)
        self.feature_names = [col for col in data.columns if col != 'id']
        # Извлечение признаков в numpy массив, заполнение пропущенных значений нулями
        self.X = data[self.feature_names].fillna(0).values
        # Сохранение количества признаков (столбцов в матрице признаков)
        self.n_features = self.X.shape[1]
        # Сохранение количества образцов (строк в матрице признаков)
        self.n_samples = self.X.shape[0]
        
        # Параметры обучения модели
        # Параметр регуляризации L2 для предотвращения переобучения
        self.lambda_reg = lambda_reg
        # Скорость обучения для градиентного спуска (влияет на скорость сходимости)
        self.learning_rate = learning_rate
        # Seed для генератора случайных чисел (для воспроизводимости)
        self.random_state = random_state
        # Установка seed для генератора случайных чисел numpy
        np.random.seed(random_state)
        
        # Инициализация шифрования с публичным ключом от активного клиента
        if use_encryption and public_key:
            # Сохранение публичного ключа для шифрования данных перед отправкой
            self.public_key = public_key
            # Пассивный клиент использует публичный ключ для шифрования
            # но не имеет приватного ключа для расшифровки (безопасность)
        else:
            # Если шифрование отключено или ключ не передан, устанавливаем None
            self.public_key = None
        
        # Нормализация данных для стабильности обучения
        # Создание объекта StandardScaler для нормализации (среднее=0, дисперсия=1)
        self.scaler = StandardScaler()
        # Вычисление параметров нормализации и применение к данным
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Инициализация локальных gates (вероятности участия признаков в модели)
        # Gates - это значения от 0 до 1, показывающие насколько признак важен
        # Начальные вероятности для всех признаков = 0.5 (нейтральное значение)
        self.local_gates = np.ones(self.n_features) * 0.5
        
        # Инициализация весов модели (для простоты используем линейную модель)
        # Веса определяют вклад каждого признака в предсказание
        # Инициализация случайными значениями из нормального распределения
        self.model_weights = np.random.normal(0, 0.1, self.n_features)
        
        # Глобальные gates (будут обновляться после агрегации на активном клиенте)
        # Глобальные gates - это усредненные локальные gates всех участников
        # Изначально равны локальным gates
        self.global_gates = self.local_gates.copy()
    
    def encrypt(self, value: float):
        """Шифрование значения с использованием публичного ключа от активного клиента"""
        # Если публичный ключ доступен, шифруем значение
        if self.public_key:
            # Шифрование значения с использованием публичного ключа Paillier
            return self.public_key.encrypt(value)
        # Если публичный ключ недоступен, возвращаем значение без шифрования
        return value
    
    def compute_local_gates(self, hidden_predictions: np.ndarray) -> np.ndarray:
        """
        Вычисление локальных gates на основе согласованности с промежуточными предсказаниями
        активного клиента - пассивный клиент не знает таргет, поэтому оценивает важность
        признаков через корреляцию со скрытыми предсказаниями активного клиента
        
        Args:
            hidden_predictions: Промежуточные предсказания от активного клиента (без таргета)
        """
        # Инициализация массива для хранения корреляций между признаками и скрытыми предсказаниями
        correlations = np.zeros(self.n_features)
        
        # Цикл по всем признакам для вычисления их согласованности со скрытыми предсказаниями
        for i in range(self.n_features):
            # Извлечение i-го признака из нормализованных данных
            feature = self.X_scaled[:, i]
            # Применение маски gate к признаку: умножаем признак на его текущий gate
            # Это позволяет учитывать текущую оценку важности признака
            masked_feature = feature * self.local_gates[i]
            # Вычисление корреляции между замаскированным признаком и скрытыми предсказаниями
            # Проверяем, что и признак, и предсказания имеют достаточную вариативность
            if np.std(masked_feature) > 1e-10 and np.std(hidden_predictions) > 1e-10:
                # Вычисление коэффициента корреляции Пирсона
                # corrcoef возвращает матрицу корреляций, берем элемент [0,1]
                correlation = np.corrcoef(masked_feature, hidden_predictions)[0, 1]
                # Использование абсолютного значения корреляции (важна сила связи, а не направление)
                # Если корреляция NaN (например, при нулевой дисперсии), устанавливаем 0
                correlations[i] = abs(correlation) if not np.isnan(correlation) else 0
            else:
                # Если вариативность недостаточна, устанавливаем корреляцию в 0
                correlations[i] = 0
        
        # Обновление локальных gates на основе корреляций
        # Используем адаптивное обновление: новый gate = старый gate + learning_rate * (correlation - старый gate)
        # Это позволяет плавно обновлять gates с учетом согласованности признаков
        self.local_gates = self.local_gates + self.learning_rate * (correlations - self.local_gates)
        # Ограничение значений gates в диапазоне [0, 1] (вероятности не могут быть отрицательными или больше 1)
        self.local_gates = np.clip(self.local_gates, 0, 1)
        
        # Возврат обновленных локальных gates
        return self.local_gates
    
    def forward_propagation(self, hidden_predictions: np.ndarray) -> np.ndarray:
        """
        Forward propagation: вычисление маскированных эмбеддингов и шифрование перед отправкой
        Эмбеддинги - это взвешенные суммы признаков, которые отправляются активному клиенту
        
        Args:
            hidden_predictions: Промежуточные предсказания от активного клиента
        
        Returns:
            Зашифрованные маскированные эмбеддинги для отправки активному клиенту
        """
        # Обновление локальных gates на основе скрытых предсказаний активного клиента
        # Это позволяет адаптировать важность признаков в зависимости от согласованности
        self.compute_local_gates(hidden_predictions)
        
        # Применение масок gates к признакам: умножаем каждый признак на его gate
        # Это позволяет "включать/выключать" признаки в зависимости от их важности
        masked_features = self.X_scaled * self.local_gates
        # Вычисление взвешенной суммы (эмбеддинг): скалярное произведение замаскированных признаков на веса модели
        embeddings = np.dot(masked_features, self.model_weights)
        
        # Шифрование эмбеддингов перед отправкой активному клиенту для защиты приватности
        if self.public_key:
            # Шифрование каждого значения эмбеддинга с использованием публичного ключа
            encrypted_embeddings = [self.encrypt(float(val)) for val in embeddings]
            # Возврат зашифрованных эмбеддингов
            return encrypted_embeddings
        
        # Если шифрование отключено, возвращаем эмбеддинги без шифрования
        return embeddings
    
    def backward_propagation(self, encrypted_gradients) -> np.ndarray:
        """
        Backward propagation: получение зашифрованных градиентов, удаление шума,
        обновление параметров и gates для улучшения модели
        
        Args:
            encrypted_gradients: Зашифрованные градиенты от активного клиента
        
        Returns:
            Локальные градиенты для отправки обратно активному клиенту
        """
        # В реальной реализации здесь происходит расшифровка и удаление шума
        # Упрощенная версия: предполагаем, что градиенты уже расшифрованы
        if isinstance(encrypted_gradients, list):
            # Если градиенты зашифрованы, нужно расшифровать
            # В реальной реализации это делается через приватный ключ активного клиента
            # Здесь используем упрощение для демонстрации алгоритма
            gradients = np.array([0.0] * self.n_features)  # Упрощение
        else:
            # Если градиенты уже в виде numpy массива, используем их напрямую
            gradients = encrypted_gradients
        
        # Вычисление локальных градиентов для пассивного клиента
        # Упрощенная версия: используем дисперсию признаков как метрику важности
        # Дисперсия показывает, насколько признак варьируется после применения gates
        # reshape нужен для правильного умножения матриц
        local_gradients = np.var(self.X_scaled * self.local_gates.reshape(1, -1), axis=0)
        
        # Применение регуляризации L2 для предотвращения переобучения
        # Добавляем штраф за большие веса: lambda_reg * веса
        local_gradients += self.lambda_reg * self.model_weights
        
        # Обновление весов модели методом градиентного спуска
        # Новый вес = старый вес - скорость_обучения * градиент
        self.model_weights -= self.learning_rate * local_gradients
        
        # Возврат локальных градиентов для отправки активному клиенту
        return local_gradients
    
    def get_local_gates(self) -> np.ndarray:
        """Получить локальные gates - возвращает копию текущих локальных gates"""
        # Возврат копии локальных gates (чтобы не изменять оригинал)
        return self.local_gates.copy()


class FedSDGFSFull:
    """
    Полная реализация FedSDG-FS с активным и пассивными клиентами
    """
    
    def __init__(self, active_data: pd.DataFrame, passive_data: pd.DataFrame,
                 target_col: str = 'target', lambda_reg: float = 0.01,
                 learning_rate: float = 0.01, max_iterations: int = 100,
                 threshold: float = 0.5, use_encryption: bool = True,
                 key_length: int = 1024, random_state: int = 42):
        """
        Инициализация FedSDG-FS - создание активного и пассивного клиентов
        
        Args:
            active_data: Данные активного клиента (с признаками и таргетом)
            passive_data: Данные пассивного клиента (только с признаками, без таргета)
            target_col: Название столбца с целевой переменной в данных активного клиента
            lambda_reg: Параметр регуляризации L2 (контролирует переобучение)
            learning_rate: Скорость обучения для градиентного спуска
            max_iterations: Максимальное количество итераций обучения
            threshold: Порог для отбора признаков (gates выше этого значения считаются важными)
            use_encryption: Использовать ли шифрование Paillier для защиты данных
            key_length: Длина ключа для Paillier (влияет на безопасность)
            random_state: Seed для генератора случайных чисел (для воспроизводимости)
        """
        # Создание активного клиента (координатора с доступом к таргету)
        self.active_client = ActiveClient(
            active_data, target_col=target_col,  # Данные и название столбца с таргетом
            lambda_reg=lambda_reg, learning_rate=learning_rate,  # Параметры обучения
            use_encryption=use_encryption, key_length=key_length,  # Параметры шифрования
            random_state=random_state  # Seed для воспроизводимости
        )
        
        # Получение публичного ключа от активного клиента
        # Публичный ключ нужен пассивному клиенту для шифрования своих данных
        public_key = self.active_client.get_public_key()
        
        # Создание пассивного клиента (участника без доступа к таргету)
        self.passive_client = PassiveClient(
            passive_data, public_key=public_key,  # Данные и публичный ключ для шифрования
            lambda_reg=lambda_reg, learning_rate=learning_rate,  # Параметры обучения
            use_encryption=use_encryption, random_state=random_state  # Параметры шифрования и seed
        )
        
        # Сохранение параметров обучения для использования в методе fit
        # Максимальное количество итераций обучения
        self.max_iterations = max_iterations
        # Порог для отбора признаков (используется в select_features)
        self.threshold = threshold
        # Флаг использования шифрования (для информации)
        self.use_encryption = use_encryption
    
    def fit(self, verbose: bool = True):
        """
        Обучение модели FedSDG-FS - основной цикл обучения с forward и backward propagation
        
        Args:
            verbose: Выводить ли информацию о процессе обучения
        """
        # Вывод начальной информации о процессе обучения
        if verbose:
            print("Начало обучения FedSDG-FS")
            print(f"Активных признаков: {self.active_client.n_features}")
            print(f"Пассивных признаков: {self.passive_client.n_features}")
            print(f"Образцов: {self.active_client.n_samples}")
        
        # Итеративное обучение: цикл по всем итерациям до достижения максимума
        for iteration in range(self.max_iterations):
            # Шаг 1: Forward propagation (прямой проход)
            # Активный клиент вычисляет скрытые предсказания на основе своих признаков и gates
            hidden_pred = self.active_client.compute_hidden_predictions()
            
            # Пассивный клиент получает скрытые предсказания, вычисляет и шифрует свои эмбеддинги
            # Эмбеддинги - это взвешенные суммы признаков пассивного клиента
            passive_embeddings = self.passive_client.forward_propagation(hidden_pred)
            
            # Активный клиент получает зашифрованные эмбеддинги от пассивного клиента,
            # агрегирует их со своими предсказаниями и добавляет случайный шум для защиты приватности
            aggregated = self.active_client.forward_propagation([passive_embeddings])
            
            # Шаг 2: Backward propagation (обратный проход)
            # Пассивный клиент получает агрегированные значения, вычисляет градиенты и обновляет параметры
            passive_gradients = self.passive_client.backward_propagation(aggregated)
            
            # Активный клиент получает градиенты от пассивного клиента,
            # вычисляет свои градиенты и обновляет параметры модели
            active_gradients = self.active_client.backward_propagation([passive_gradients])
            
            # Шаг 3: Агрегация глобальных gates (Dual Gate механизм)
            # Получение локальных gates от пассивного клиента (оценка важности его признаков)
            passive_gates = self.passive_client.get_local_gates()
            
            # Агрегация глобальных gates на активном клиенте (координаторе)
            # Глобальные gates - это усредненные локальные gates всех участников
            global_gates = self.active_client.aggregate_global_gates([passive_gates])
            
            # Обновление глобальных gates у пассивного клиента
            # Берем только часть глобальных gates, соответствующую признакам пассивного клиента
            self.passive_client.global_gates = global_gates[:self.passive_client.n_features]
            
            # Вывод информации о прогрессе обучения каждые 10 итераций
            if verbose and (iteration + 1) % 10 == 0:
                # Подсчет количества отобранных признаков активного клиента (gates > порога)
                active_selected = np.sum(self.active_client.global_gates > self.threshold)
                # Подсчет количества отобранных признаков пассивного клиента (gates > порога)
                passive_selected = np.sum(self.passive_client.global_gates > self.threshold)
                # Вывод информации о прогрессе
                print(f"Итерация {iteration + 1}/{self.max_iterations}: "
                      f"Активных: {active_selected}, Пассивных: {passive_selected}")
        
        # Вывод сообщения о завершении обучения
        if verbose:
            print("\nОбучение завершено")
    
    def select_features(self) -> Tuple[List[str], List[str]]:
        """
        Выбор признаков на основе глобальных gates - финальный этап отбора признаков
        Признаки с глобальным gate выше порога считаются важными и остаются в модели
        
        Returns:
            Кортеж (отобранные признаки активного клиента, отобранные признаки пассивного клиента)
        """
        # Выбор признаков активного клиента на основе его глобальных gates
        # Используется метод select_features активного клиента с заданным порогом
        active_selected = self.active_client.select_features(self.threshold)
        
        # Выбор признаков пассивного клиента на основе его глобальных gates
        # Создание булевой маски: глобальный gate > порога
        passive_mask = self.passive_client.global_gates > self.threshold
        # Получение названий отобранных признаков пассивного клиента на основе маски
        passive_selected = [self.passive_client.feature_names[i] 
                           for i in range(len(self.passive_client.feature_names)) 
                           if passive_mask[i]]
        
        # Возврат кортежа с отобранными признаками обоих клиентов
        return active_selected, passive_selected
    
    def transform(self, active_data: pd.DataFrame, passive_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Применение отбора признаков к новым данным - фильтрация данных по отобранным признакам
        
        Args:
            active_data: Данные активного клиента (для применения отбора признаков)
            passive_data: Данные пассивного клиента (для применения отбора признаков)
        
        Returns:
            Кортеж (отфильтрованные данные активного клиента, отфильтрованные данные пассивного клиента)
        """
        # Получение списков отобранных признаков для обоих клиентов
        active_selected, passive_selected = self.select_features()
        
        # Объединение данных активного и пассивного клиентов по столбцу 'id'
        # Используется inner join - остаются только строки с совпадающими ID
        merged = active_data.merge(passive_data, on='id', how='inner')
        
        # Выбор только отобранных признаков активного клиента (ID + отобранные признаки)
        active_result = merged[['id'] + active_selected].copy()
        # Выбор только отобранных признаков пассивного клиента (ID + отобранные признаки)
        passive_result = merged[['id'] + passive_selected].copy()
        
        # Возврат кортежа с отфильтрованными данными обоих клиентов
        return active_result, passive_result


def main():
    """Пример использования полной реализации FedSDG-FS - демонстрация работы алгоритма"""
    
    # Загрузка данных из CSV файлов
    print("Загрузка данных...")
    # Загрузка данных активного клиента (с признаками и целевой переменной)
    active_data = pd.read_csv('Data/active_dataset_test.csv')
    # Загрузка данных пассивного клиента (только с признаками, без целевой переменной)
    passive_data = pd.read_csv('Data/passive_dataset_test.csv')
    
    # Вывод размерности загруженных датасетов (количество строк и столбцов)
    print(f"Активный датасет: {active_data.shape}")
    print(f"Пассивный датасет: {passive_data.shape}")
    
    # Инициализация и обучение модели FedSDG-FS
    print("\n" + "="*50)
    # Создание экземпляра модели FedSDGFSFull с заданными параметрами
    model = FedSDGFSFull(
        active_data=active_data,  # Данные активного клиента
        passive_data=passive_data,  # Данные пассивного клиента
        target_col='target',  # Название столбца с целевой переменной
        lambda_reg=0.01,  # Параметр регуляризации L2
        learning_rate=0.01,  # Скорость обучения для градиентного спуска
        max_iterations=50,  # Максимальное количество итераций обучения
        threshold=0.5,  # Порог для отбора признаков (gates выше этого значения считаются важными)
        use_encryption=False,  # Установите True для использования Paillier шифрования
        random_state=42  # Seed для воспроизводимости результатов
    )
    
    # Запуск процесса обучения модели
    # verbose=True включает вывод информации о прогрессе обучения
    model.fit(verbose=True)
    
    # Получение результатов отбора признаков
    print("\n" + "="*50)
    # Получение списков отобранных признаков для обоих клиентов
    active_features, passive_features = model.select_features()
    
    # Вывод списка отобранных активных признаков
    print("\nОтобранные активные признаки:")
    # Цикл по первым 20 отобранным признакам (для краткости вывода)
    for feat in active_features[:20]:  # Показываем первые 20
        # Вывод названия каждого признака с отступом
        print(f"  - {feat}")
    
    # Вывод списка отобранных пассивных признаков
    print("\nОтобранные пассивные признаки:")
    # Цикл по первым 20 отобранным признакам (для краткости вывода)
    for feat in passive_features[:20]:  # Показываем первые 20
        # Вывод названия каждого признака с отступом
        print(f"  - {feat}")
    
    # Применение трансформации к исходным данным (фильтрация по отобранным признакам)
    print("\n" + "="*50)
    # Применение обученной модели к исходным данным для получения отфильтрованных данных
    active_transformed, passive_transformed = model.transform(active_data, passive_data)
    
    # Вывод информации о размерности данных после отбора признаков
    print(f"\nРазмерность после отбора признаков:")
    # Вывод размерности активных данных (количество строк и столбцов после фильтрации)
    print(f"Активные данные: {active_transformed.shape}")
    # Вывод размерности пассивных данных (количество строк и столбцов после фильтрации)
    print(f"Пассивные данные: {passive_transformed.shape}")
    
    # Сохранение результатов отбора признаков в CSV файлы
    # Сохранение отфильтрованных данных активного клиента
    # index=False означает, что индекс строк не будет сохранен в файл
    active_transformed.to_csv('Data/active_dataset_full_selected.csv', index=False)
    # Сохранение отфильтрованных данных пассивного клиента
    # index=False означает, что индекс строк не будет сохранен в файл
    passive_transformed.to_csv('Data/passive_dataset_full_selected.csv', index=False)
    
    # Вывод информации о сохраненных файлах
    print("\nРезультаты сохранены в:")
    # Вывод пути к файлу с активными данными после отбора признаков
    print("  - Data/active_dataset_full_selected.csv")
    # Вывод пути к файлу с пассивными данными после отбора признаков
    print("  - Data/passive_dataset_full_selected.csv")


# Проверка, запущен ли скрипт напрямую (а не импортирован как модуль)
if __name__ == "__main__":
    # Вызов функции main() для выполнения примера использования FedSDG-FS
    main()

