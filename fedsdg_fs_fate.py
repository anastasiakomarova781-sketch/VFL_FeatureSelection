"""
FedSDG-FS для интеграции с FATE Framework
Версия для вертикального федеративного обучения с поддержкой Paillier шифрования
"""

# Импорт библиотеки numpy для работы с массивами и математическими операциями
import numpy as np
# Импорт библиотеки pandas для работы с табличными данными (DataFrame)
import pandas as pd
# Импорт типов данных для аннотации типов в функциях
from typing import List, Tuple, Dict, Optional
# Импорт StandardScaler для нормализации данных (приведение к среднему 0 и дисперсии 1)
from sklearn.preprocessing import StandardScaler
# Импорт модуля warnings для управления предупреждениями
import warnings
# Отключение всех предупреждений для чистого вывода
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

# Для интеграции с FATE (если установлен)
# Попытка импортировать модули из FATE Framework
try:
    # Импорт класса EncryptParam для параметров шифрования из FATE
    from federatedml.param import EncryptParam
    # Импорт модуля consts с константами из FATE
    from federatedml.util import consts
    # Установка флага доступности FATE в True при успешном импорте
    FATE_AVAILABLE = True
except ImportError:
    # Если импорт не удался (FATE не установлен), устанавливаем флаг в False
    FATE_AVAILABLE = False


class PaillierEncryptionFATE:
    """
    Класс шифрования Paillier совместимый с FATE
    """
    
    def __init__(self, encrypt_param: Optional[object] = None, key_length: int = 1024):
        """
        Инициализация шифрования
        
        Args:
            encrypt_param: Параметры шифрования из FATE (EncryptParam)
            key_length: Длина ключа в битах
        """
        # Проверка доступности библиотеки Paillier
        if not PAILLIER_AVAILABLE:
            # Выброс исключения, если библиотека не установлена
            raise ImportError("phe library is required. Install with: pip install phe")
        
        # Если передан параметр шифрования из FATE и FATE доступен
        if encrypt_param and FATE_AVAILABLE:
            # Используем длину ключа из параметров FATE, если она есть
            # Иначе используем переданное значение key_length
            key_length = encrypt_param.key_length if hasattr(encrypt_param, 'key_length') else key_length
        
        # Генерация пары ключей (публичный и приватный) для шифрования Paillier
        # Публичный ключ используется для шифрования, приватный - для расшифровки
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        # Сохранение длины ключа в атрибуте объекта
        self.key_length = key_length
    
    def encrypt(self, value: float) -> paillier.EncryptedNumber:
        """Шифрование значения"""
        # Шифрование числового значения с использованием публичного ключа
        # Возвращает зашифрованное число типа EncryptedNumber
        return self.public_key.encrypt(value)
    
    def encrypt_matrix(self, matrix: np.ndarray) -> List[List]:
        """Шифрование матрицы значений"""
        # Инициализация списка для хранения зашифрованной матрицы
        encrypted_matrix = []
        # Цикл по каждой строке матрицы
        for row in matrix:
            # Инициализация списка для хранения зашифрованной строки
            encrypted_row = []
            # Цикл по каждому значению в строке
            for val in row:
                # Проверка, что значение не является NaN
                if not np.isnan(val):
                    # Шифрование значения и добавление в зашифрованную строку
                    encrypted_row.append(self.encrypt(float(val)))
                else:
                    # Если значение NaN, добавляем None вместо зашифрованного значения
                    encrypted_row.append(None)
            # Добавление зашифрованной строки в матрицу
            encrypted_matrix.append(encrypted_row)
        # Возврат зашифрованной матрицы в виде списка списков
        return encrypted_matrix
    
    def decrypt(self, encrypted_value: paillier.EncryptedNumber) -> float:
        """Расшифровка значения"""
        # Расшифровка зашифрованного значения с использованием приватного ключа
        # Возвращает исходное числовое значение типа float
        return self.private_key.decrypt(encrypted_value)
    
    def decrypt_matrix(self, encrypted_matrix: List[List]) -> np.ndarray:
        """Расшифровка матрицы значений"""
        # Инициализация списка для хранения расшифрованной матрицы
        decrypted = []
        # Цикл по каждой строке зашифрованной матрицы
        for row in encrypted_matrix:
            # Инициализация списка для хранения расшифрованной строки
            decrypted_row = []
            # Цикл по каждому значению в строке
            for val in row:
                # Проверка, что значение не является None
                if val is not None:
                    # Расшифровка значения и добавление в расшифрованную строку
                    decrypted_row.append(self.decrypt(val))
                else:
                    # Если значение None, добавляем NaN вместо расшифрованного значения
                    decrypted_row.append(np.nan)
            # Добавление расшифрованной строки в матрицу
            decrypted.append(decrypted_row)
        # Преобразование списка списков в numpy массив и возврат
        return np.array(decrypted)


class FedSDGFSFATE:
    """
    FedSDG-FS для вертикального федеративного обучения в FATE
    Поддерживает работу в режиме активной и пассивной стороны
    """
    
    def __init__(
        self,
        role: str = 'active',  # 'active' или 'passive'
        lambda_reg: float = 0.01,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        threshold: float = 0.5,
        use_encryption: bool = True,
        encrypt_param: Optional[object] = None,
        key_length: int = 1024,
        random_state: int = 42
    ):
        """
        Инициализация FedSDG-FS для FATE
        
        Args:
            role: Роль участника ('active' или 'passive')
            lambda_reg: Параметр регуляризации
            learning_rate: Скорость обучения
            max_iterations: Максимальное количество итераций
            threshold: Порог для отбора признаков
            use_encryption: Использовать ли шифрование Paillier
            encrypt_param: Параметры шифрования из FATE
            key_length: Длина ключа для Paillier
            random_state: Seed для воспроизводимости
        """
        # Сохранение роли участника ('active' - активная сторона с метками, 'passive' - пассивная без меток)
        self.role = role
        # Сохранение параметра регуляризации L2 (контролирует переобучение)
        self.lambda_reg = lambda_reg
        # Сохранение скорости обучения для градиентного спуска
        self.learning_rate = learning_rate
        # Сохранение максимального количества итераций обучения
        self.max_iterations = max_iterations
        # Сохранение порогового значения для отбора признаков (гейты выше порога считаются активными)
        self.threshold = threshold
        # Сохранение флага использования шифрования
        self.use_encryption = use_encryption
        # Сохранение seed для генератора случайных чисел (для воспроизводимости результатов)
        self.random_state = random_state
        
        # Если шифрование включено, создаем объект PaillierEncryptionFATE
        if use_encryption:
            # Инициализация объекта шифрования с параметрами из FATE или заданной длиной ключа
            self.encryption = PaillierEncryptionFATE(encrypt_param=encrypt_param, key_length=key_length)
        else:
            # Если шифрование отключено, устанавливаем encryption в None
            self.encryption = None
        
        # Инициализация пустого списка для хранения отобранных признаков
        self.selected_features = []
        # Инициализация весов гейтов (пока None, будет установлено при обучении)
        self.gate_weights = None
        # Создание объекта StandardScaler для нормализации данных
        self.scaler = StandardScaler()
        # Инициализация пустого списка для хранения названий признаков
        self.feature_names = []
        
        # Установка seed для генератора случайных чисел numpy (для воспроизводимости)
        np.random.seed(random_state)
    
    def _prepare_local_data(
        self,
        data: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Подготовка локальных данных
        
        Args:
            data: Локальные данные
            target_col: Название столбца с целевой переменной (только для active)
        
        Returns:
            X, y (или None для passive), feature_names
        """
        # Извлечение признаков
        # Если роль - активная сторона и указана целевая переменная
        if self.role == 'active' and target_col:
            # Извлечение списка названий признаков (исключаем 'id' и целевую переменную)
            feature_cols = [col for col in data.columns if col not in ['id', target_col]]
            # Сохранение названий признаков в атрибуте объекта
            self.feature_names = feature_cols
            # Преобразование признаков в numpy массив
            X = data[feature_cols].values
            # Извлечение целевой переменной, если она есть в данных
            y = data[target_col].values if target_col in data.columns else None
        else:
            # Для пассивной стороны или если целевая переменная не указана
            # Извлечение списка названий признаков (исключаем только 'id')
            feature_cols = [col for col in data.columns if col != 'id']
            # Сохранение названий признаков в атрибуте объекта
            self.feature_names = feature_cols
            # Преобразование признаков в numpy массив
            X = data[feature_cols].values
            # Для пассивной стороны целевая переменная отсутствует
            y = None
        
        # Обработка пропущенных значений
        # Заполняем пропуски нулями и преобразуем обратно в numpy массив
        X = pd.DataFrame(X).fillna(0).values
        
        # Возврат подготовленных данных: признаки, целевая переменная (или None), названия признаков
        return X, y, self.feature_names
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Сигмоидная функция"""
        # Применение сигмоидной функции: 1 / (1 + exp(-x))
        # Используется clip для ограничения значений в диапазоне [-500, 500]
        # чтобы избежать переполнения при вычислении экспоненты
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _compute_local_gate_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление локальных оценок гейтов
        
        Args:
            X: Матрица признаков
        
        Returns:
            Оценки гейтов
        """
        # Вычисление скалярного произведения матрицы признаков на веса гейтов
        scores = np.dot(X, self.gate_weights)
        # Применение сигмоидной функции к полученным оценкам
        # Результат - значения в диапазоне [0, 1], которые показывают "открытость" гейта
        return self._sigmoid(scores)
    
    def initialize(self, n_features: int):
        """
        Инициализация весов гейтов
        
        Args:
            n_features: Количество признаков
        """
        # Инициализация весов гейтов случайными значениями из нормального распределения
        # Со средним 0 и стандартным отклонением 0.1
        self.gate_weights = np.random.normal(0, 0.1, n_features)
    
    def compute_gate_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление оценок гейтов (публичный метод для FATE)
        
        Args:
            X: Матрица признаков
        
        Returns:
            Оценки гейтов
        """
        # Проверка, инициализированы ли веса гейтов
        if self.gate_weights is None:
            # Если веса не инициализированы, инициализируем их на основе количества признаков
            self.initialize(X.shape[1])
        
        # Вызов приватного метода для вычисления оценок гейтов
        return self._compute_local_gate_scores(X)
    
    def update_gate_weights(
        self,
        X: np.ndarray,
        importance_scores: np.ndarray
    ):
        """
        Обновление весов гейтов
        
        Args:
            X: Матрица признаков
            importance_scores: Оценки важности признаков
        """
        # Проверка, инициализированы ли веса гейтов
        if self.gate_weights is None:
            # Если веса не инициализированы, инициализируем их на основе количества признаков
            self.initialize(X.shape[1])
        
        # Вычисление градиента
        # Градиент = важность признаков минус регуляризация (L2)
        gradient = importance_scores - self.lambda_reg * self.gate_weights
        
        # Обновление весов
        # Новый вес = старый вес + скорость_обучения * градиент
        self.gate_weights += self.learning_rate * gradient
        
        # Применение регуляризации: ограничение весов в диапазоне [-1.0, 1.0]
        # Это предотвращает слишком большие значения весов
        self.gate_weights = np.clip(self.gate_weights, -1.0, 1.0)
    
    def select_features(self, X: np.ndarray) -> List[int]:
        """
        Отбор признаков на основе финальных оценок гейтов
        
        Args:
            X: Матрица признаков
        
        Returns:
            Индексы отобранных признаков
        """
        # Вычисление оценок гейтов для всех признаков
        gate_scores = self.compute_gate_scores(X)
        # Вычисление средних значений оценок гейтов по всем образцам (по оси 0)
        mean_scores = gate_scores.mean(axis=0)
        # Нахождение индексов признаков, у которых средняя оценка гейта выше порога
        # np.where возвращает кортеж массивов, берем первый элемент [0] и преобразуем в список
        selected_indices = np.where(mean_scores > self.threshold)[0].tolist()
        
        # Формирование списка названий отобранных признаков на основе их индексов
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        # Возврат списка индексов отобранных признаков
        return selected_indices
    
    def fit_local(
        self,
        data: pd.DataFrame,
        target_col: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Локальное обучение (для тестирования или standalone режима)
        
        Args:
            data: Локальные данные
            target_col: Название столбца с целевой переменной
            verbose: Выводить ли информацию
        """
        # Подготовка локальных данных: извлечение признаков, целевой переменной и названий
        X, y, feature_names = self._prepare_local_data(data, target_col)
        
        # Нормализация
        # Нормализация данных (приведение к среднему 0 и дисперсии 1)
        # fit_transform вычисляет параметры нормализации и применяет их
        X_scaled = self.scaler.fit_transform(X)
        
        # Инициализация
        # Инициализация весов гейтов на основе количества признаков
        self.initialize(X_scaled.shape[1])
        
        # Если включен режим подробного вывода информации
        if verbose:
            # Вывод информации о начале локального обучения с указанием роли
            print(f"Локальное обучение ({self.role} сторона)")
            # Вывод количества признаков и образцов
            print(f"Признаков: {X_scaled.shape[1]}, Образцов: {X_scaled.shape[0]}")
        
        # Итеративное обучение: цикл по всем итерациям до достижения максимума
        for iteration in range(self.max_iterations):
            # Вычисление оценок гейтов для всех признаков
            gate_scores = self.compute_gate_scores(X_scaled)
            
            # Вычисление важности признаков
            # Если роль - активная сторона и есть целевая переменная
            if self.role == 'active' and y is not None:
                # Инициализация массива важности признаков нулями
                importance = np.zeros(X_scaled.shape[1])
                # Цикл по всем признакам
                for i in range(X_scaled.shape[1]):
                    # Проверка, что признак имеет достаточную вариативность (не константа)
                    if np.std(X_scaled[:, i]) > 1e-10:
                        # Применение маски гейта к признаку (умножение признака на оценку гейта)
                        masked_feature = X_scaled[:, i] * gate_scores[:, i]
                        # Вычисление корреляции между замаскированным признаком и целевой переменной
                        # corrcoef возвращает матрицу корреляций, берем элемент [0,1]
                        correlation = np.corrcoef(masked_feature, y)[0, 1]
                        # Сохранение абсолютного значения корреляции как важности признака
                        # Если корреляция NaN, устанавливаем 0
                        importance[i] = abs(correlation) if not np.isnan(correlation) else 0
            else:
                # Для пассивной стороны используем дисперсию
                # Вычисление дисперсии замаскированных признаков (признак * гейт) по образцам
                importance = np.var(X_scaled * gate_scores, axis=0)
            
            # Обновление весов
            # Обновление весов гейтов на основе вычисленной важности признаков
            self.update_gate_weights(X_scaled, importance)
            
            # Если включен подробный вывод и текущая итерация кратна 10
            if verbose and (iteration + 1) % 10 == 0:
                # Подсчет количества отобранных признаков (гейты со средним значением выше порога)
                selected = np.sum(gate_scores.mean(axis=0) > self.threshold)
                # Вывод информации о прогрессе обучения
                print(f"Итерация {iteration + 1}/{self.max_iterations}: "
                      f"Отобрано признаков: {selected}")
        
        # Финальный отбор
        # Выполнение финального отбора признаков на основе обученных весов гейтов
        selected_indices = self.select_features(X_scaled)
        
        # Если включен подробный вывод информации
        if verbose:
            # Вывод информации о завершении отбора признаков
            print(f"\nОтбор завершен: выбрано {len(selected_indices)} признаков")
    
    def transform_local(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применение отбора признаков к локальным данным
        
        Args:
            data: Локальные данные
        
        Returns:
            Отфильтрованные данные
        """
        # Формирование списка столбцов для выборки: ID + отобранные признаки
        selected_cols = ['id'] + self.selected_features
        # Извлечение отобранных столбцов из данных и создание независимой копии
        return data[selected_cols].copy()
    
    def get_selected_features(self) -> List[str]:
        """Получить список отобранных признаков"""
        # Возврат списка названий отобранных признаков
        return self.selected_features
    
    def get_public_key(self):
        """Получить публичный ключ для шифрования (для передачи другой стороне)"""
        # Проверка, инициализировано ли шифрование
        if self.encryption:
            # Возврат публичного ключа для передачи другой стороне в федеративном обучении
            return self.encryption.public_key
        # Если шифрование не используется, возвращаем None
        return None


def run_fate_integration_example():
    """
    Пример использования для интеграции с FATE
    Демонстрирует работу активной и пассивной сторон
    """
    
    # Загрузка данных
    # Вывод сообщения о начале загрузки данных
    print("Загрузка данных...")
    # Загрузка данных активной стороны из CSV файла
    active_data = pd.read_csv('Data/active_dataset_test.csv')
    # Загрузка данных пассивной стороны из CSV файла
    passive_data = pd.read_csv('Data/passive_dataset_test.csv')
    
    # Активная сторона
    # Вывод разделительной линии для визуального оформления
    print("\n" + "="*50)
    # Вывод заголовка для активной стороны
    print("АКТИВНАЯ СТОРОНА")
    # Вывод разделительной линии для визуального оформления
    print("="*50)
    
    # Создание экземпляра модели FedSDGFSFATE для активной стороны
    active_model = FedSDGFSFATE(
        role='active',  # Роль: активная сторона (имеет доступ к целевой переменной)
        lambda_reg=0.01,  # Параметр регуляризации L2
        learning_rate=0.01,  # Скорость обучения для градиентного спуска
        max_iterations=50,  # Максимальное количество итераций обучения
        threshold=0.3,  # Порог для отбора признаков
        use_encryption=True,  # Включение шифрования Paillier
        random_state=42  # Seed для воспроизводимости результатов
    )
    
    # Обучение модели активной стороны на локальных данных
    # verbose=True включает вывод информации о процессе обучения
    active_model.fit_local(active_data, target_col='target', verbose=True)
    # Получение списка отобранных признаков активной стороны
    active_selected = active_model.get_selected_features()
    
    # Вывод информации об отобранных признаках активной стороны
    print(f"\nОтобранные признаки активной стороны ({len(active_selected)}):")
    # Цикл по первым 10 отобранным признакам (для краткости вывода)
    for feat in active_selected[:10]:  # Показываем первые 10
        # Вывод названия каждого признака с отступом
        print(f"  - {feat}")
    
    # Пассивная сторона
    # Вывод разделительной линии для визуального оформления
    print("\n" + "="*50)
    # Вывод заголовка для пассивной стороны
    print("ПАССИВНАЯ СТОРОНА")
    # Вывод разделительной линии для визуального оформления
    print("="*50)
    
    # Создание экземпляра модели FedSDGFSFATE для пассивной стороны
    passive_model = FedSDGFSFATE(
        role='passive',  # Роль: пассивная сторона (не имеет доступа к целевой переменной)
        lambda_reg=0.01,  # Параметр регуляризации L2
        learning_rate=0.01,  # Скорость обучения для градиентного спуска
        max_iterations=50,  # Максимальное количество итераций обучения
        threshold=0.3,  # Порог для отбора признаков
        use_encryption=True,  # Включение шифрования Paillier
        random_state=42  # Seed для воспроизводимости результатов
    )
    
    # Обучение модели пассивной стороны на локальных данных
    # verbose=True включает вывод информации о процессе обучения
    passive_model.fit_local(passive_data, verbose=True)
    # Получение списка отобранных признаков пассивной стороны
    passive_selected = passive_model.get_selected_features()
    
    # Вывод информации об отобранных признаках пассивной стороны
    print(f"\nОтобранные признаки пассивной стороны ({len(passive_selected)}):")
    # Цикл по первым 10 отобранным признакам (для краткости вывода)
    for feat in passive_selected[:10]:  # Показываем первые 10
        # Вывод названия каждого признака с отступом
        print(f"  - {feat}")
    
    # Применение трансформации
    # Применение обученной модели активной стороны к исходным данным
    active_transformed = active_model.transform_local(active_data)
    # Применение обученной модели пассивной стороны к исходным данным
    passive_transformed = passive_model.transform_local(passive_data)
    
    # Вывод информации о размерности данных после отбора признаков
    print(f"\nРезультаты:")
    # Вывод размерности активных данных (количество строк и столбцов)
    print(f"Активные данные: {active_transformed.shape}")
    # Вывод размерности пассивных данных (количество строк и столбцов)
    print(f"Пассивные данные: {passive_transformed.shape}")
    
    # Сохранение результатов
    # Сохранение отфильтрованных данных активной стороны в CSV файл
    # index=False означает, что индекс строк не будет сохранен в файл
    active_transformed.to_csv('Data/active_dataset_fate_selected.csv', index=False)
    # Сохранение отфильтрованных данных пассивной стороны в CSV файл
    # index=False означает, что индекс строк не будет сохранен в файл
    passive_transformed.to_csv('Data/passive_dataset_fate_selected.csv', index=False)
    
    # Вывод информации о сохраненных файлах
    print("\nРезультаты сохранены в:")
    # Вывод пути к файлу с активными данными
    print("  - Data/active_dataset_fate_selected.csv")
    # Вывод пути к файлу с пассивными данными
    print("  - Data/passive_dataset_fate_selected.csv")


# Проверка, запущен ли скрипт напрямую (а не импортирован как модуль)
if __name__ == "__main__":
    # Вызов функции run_fate_integration_example() для выполнения примера использования
    run_fate_integration_example()

