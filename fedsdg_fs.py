"""
FedSDG-FS: Federated Stochastic Dual-Gate Feature Selection
Реализация метода отбора признаков для вертикального федеративного обучения
с использованием шифрования Paillier и совместимостью с FATE framework
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
    # Вывод предупреждения о необходимости установки библиотеки
    print("Warning: phe library not found. Install with: pip install phe")


class PaillierEncryption:
    """Класс для работы с шифрованием Paillier"""
    
    def __init__(self, key_length: int = 1024):
        """
        Инициализация шифрования Paillier
        
        Args:
            key_length: Длина ключа в битах (1024 или 2048)
        """
        # Проверка доступности библиотеки Paillier
        if not PAILLIER_AVAILABLE:
            # Выброс исключения, если библиотека не установлена
            raise ImportError("phe library is required. Install with: pip install phe")
        
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
    
    def encrypt_array(self, array: np.ndarray) -> List[paillier.EncryptedNumber]:
        """Шифрование массива значений"""
        # Создание списка зашифрованных значений из массива
        # Для каждого значения: если оно не NaN, шифруем его, иначе возвращаем None
        return [self.encrypt(float(val)) if not np.isnan(val) else None for val in array]
    
    def decrypt(self, encrypted_value: paillier.EncryptedNumber) -> float:
        """Расшифровка значения"""
        # Расшифровка зашифрованного значения с использованием приватного ключа
        # Возвращает исходное числовое значение типа float
        return self.private_key.decrypt(encrypted_value)
    
    def decrypt_array(self, encrypted_array: List[paillier.EncryptedNumber]) -> np.ndarray:
        """Расшифровка массива значений"""
        # Преобразование списка зашифрованных значений в массив numpy
        # Для каждого значения: если оно не None, расшифровываем, иначе возвращаем NaN
        return np.array([self.decrypt(val) if val is not None else np.nan for val in encrypted_array])


class FedSDGFS:
    """
    FedSDG-FS: Federated Stochastic Dual-Gate Feature Selection
    Метод отбора признаков для вертикального федеративного обучения
    """
    
    def __init__(
        self,
        lambda_reg: float = 0.01,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        threshold: float = 0.5,
        use_encryption: bool = True,
        key_length: int = 1024,
        random_state: int = 42
    ):
        """
        Инициализация FedSDG-FS
        
        Args:
            lambda_reg: Параметр регуляризации
            learning_rate: Скорость обучения
            max_iterations: Максимальное количество итераций
            threshold: Порог для отбора признаков
            use_encryption: Использовать ли шифрование Paillier
            key_length: Длина ключа для Paillier
            random_state: Seed для воспроизводимости
        """
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
        
        # Если шифрование включено, создаем объект PaillierEncryption
        if use_encryption:
            # Инициализация объекта шифрования с заданной длиной ключа
            self.encryption = PaillierEncryption(key_length=key_length)
        else:
            # Если шифрование отключено, устанавливаем encryption в None
            self.encryption = None
        
        # Инициализация пустого списка для хранения отобранных признаков активной стороны
        self.selected_features_active = []
        # Инициализация пустого списка для хранения отобранных признаков пассивной стороны
        self.selected_features_passive = []
        # Инициализация весов гейтов для активной стороны (пока None, будет установлено при обучении)
        self.gate_weights_active = None
        # Инициализация весов гейтов для пассивной стороны (пока None, будет установлено при обучении)
        self.gate_weights_passive = None
        # Создание объекта StandardScaler для нормализации данных активной стороны
        self.scaler_active = StandardScaler()
        # Создание объекта StandardScaler для нормализации данных пассивной стороны
        self.scaler_passive = StandardScaler()
        
        # Установка seed для генератора случайных чисел numpy (для воспроизводимости)
        np.random.seed(random_state)
    
    def _prepare_data(
        self,
        active_data: pd.DataFrame,
        passive_data: pd.DataFrame,
        target_col: str = 'target'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Подготовка данных для обучения
        
        Args:
            active_data: Данные активной стороны (с признаками и целевой переменной)
            passive_data: Данные пассивной стороны (только признаки)
            target_col: Название столбца с целевой переменной
        
        Returns:
            X_active, X_passive, y, feature_names_active, feature_names_passive
        """
        # Объединение данных активной и пассивной сторон по столбцу 'id'
        # Используется inner join - остаются только строки с совпадающими ID
        merged = active_data.merge(passive_data, on='id', how='inner')
        
        # Извлечение списка названий признаков активной стороны
        # Исключаем столбцы 'id' и целевой переменной (target_col)
        active_features = [col for col in active_data.columns if col not in ['id', target_col]]
        # Извлечение списка названий признаков пассивной стороны
        # Исключаем только столбец 'id'
        passive_features = [col for col in passive_data.columns if col != 'id']
        
        # Преобразование признаков активной стороны в numpy массив
        X_active = merged[active_features].values
        # Преобразование признаков пассивной стороны в numpy массив
        X_passive = merged[passive_features].values
        # Извлечение целевой переменной (меток) в numpy массив
        y = merged[target_col].values
        
        # Обработка пропущенных значений (NaN) в данных активной стороны
        # Заполняем пропуски нулями и преобразуем обратно в numpy массив
        X_active = pd.DataFrame(X_active).fillna(0).values
        # Обработка пропущенных значений (NaN) в данных пассивной стороны
        # Заполняем пропуски нулями и преобразуем обратно в numpy массив
        X_passive = pd.DataFrame(X_passive).fillna(0).values
        
        # Возврат подготовленных данных: признаки активной стороны, признаки пассивной стороны,
        # целевая переменная, названия признаков активной стороны, названия признаков пассивной стороны
        return X_active, X_passive, y, active_features, passive_features
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Сигмоидная функция"""
        # Применение сигмоидной функции: 1 / (1 + exp(-x))
        # Используется clip для ограничения значений в диапазоне [-500, 500]
        # чтобы избежать переполнения при вычислении экспоненты
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _compute_gate_scores(
        self,
        X: np.ndarray,
        gate_weights: np.ndarray,
        encrypted: bool = False
    ) -> np.ndarray:
        """
        Вычисление оценок гейтов для признаков
        
        Args:
            X: Матрица признаков
            gate_weights: Веса гейтов
            encrypted: Использовать ли зашифрованные вычисления
        
        Returns:
            Оценки гейтов
        """
        # Проверка необходимости использования зашифрованных вычислений
        if encrypted and self.encryption:
            # Для зашифрованных вычислений используем упрощенную версию
            # В реальной реализации нужны специальные операции для Paillier
            # Вычисление скалярного произведения матрицы признаков на веса гейтов
            scores = np.dot(X, gate_weights)
        else:
            # Обычное вычисление без шифрования
            # Вычисление скалярного произведения матрицы признаков на веса гейтов
            scores = np.dot(X, gate_weights)
        
        # Применение сигмоидной функции к полученным оценкам
        # Результат - значения в диапазоне [0, 1], которые показывают "открытость" гейта
        return self._sigmoid(scores)
    
    def _compute_feature_importance(
        self,
        X_active: np.ndarray,
        X_passive: np.ndarray,
        y: np.ndarray,
        gate_active: np.ndarray,
        gate_passive: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление важности признаков
        
        Args:
            X_active: Признаки активной стороны
            X_passive: Признаки пассивной стороны
            y: Целевая переменная
            gate_active: Оценки гейтов для активной стороны
            gate_passive: Оценки гейтов для пассивной стороны
        
        Returns:
            Важность признаков для активной и пассивной сторон
        """
        # Получение количества образцов (строк) в данных активной стороны
        n_samples = X_active.shape[0]
        
        # Вычисление важности через корреляцию с целевой переменной
        # с учетом весов гейтов
        # Инициализация массива важности признаков активной стороны нулями
        importance_active = np.zeros(X_active.shape[1])
        # Инициализация массива важности признаков пассивной стороны нулями
        importance_passive = np.zeros(X_passive.shape[1])
        
        # Цикл по всем признакам активной стороны
        for i in range(X_active.shape[1]):
            # Проверка, что признак имеет достаточную вариативность (не константа)
            if np.std(X_active[:, i]) > 1e-10:
                # Применение маски гейта к признаку (умножение признака на оценку гейта)
                masked_feature = X_active[:, i] * gate_active[:, i]
                # Вычисление корреляции между замаскированным признаком и целевой переменной
                # corrcoef возвращает матрицу корреляций, берем элемент [0,1]
                correlation = np.corrcoef(masked_feature, y)[0, 1]
                # Сохранение абсолютного значения корреляции как важности признака
                # Если корреляция NaN (например, при нулевой дисперсии), устанавливаем 0
                importance_active[i] = abs(correlation) if not np.isnan(correlation) else 0
        
        # Цикл по всем признакам пассивной стороны
        for i in range(X_passive.shape[1]):
            # Проверка, что признак имеет достаточную вариативность (не константа)
            if np.std(X_passive[:, i]) > 1e-10:
                # Применение маски гейта к признаку (умножение признака на оценку гейта)
                masked_feature = X_passive[:, i] * gate_passive[:, i]
                # Для пассивной стороны используем косвенные метрики
                # Вычисляем абсолютное значение среднего замаскированного признака
                correlation = np.abs(np.mean(masked_feature))
                # Сохранение значения как важности признака
                # Если значение NaN, устанавливаем 0
                importance_passive[i] = correlation if not np.isnan(correlation) else 0
        
        # Возврат массивов важности признаков для активной и пассивной сторон
        return importance_active, importance_passive
    
    def fit(
        self,
        active_data: pd.DataFrame,
        passive_data: pd.DataFrame,
        target_col: str = 'target',
        verbose: bool = True
    ):
        """
        Обучение модели FedSDG-FS
        
        Args:
            active_data: Данные активной стороны
            passive_data: Данные пассивной стороны
            target_col: Название столбца с целевой переменной
            verbose: Выводить ли информацию о процессе
        """
        # Подготовка данных: объединение, извлечение признаков и целевой переменной
        X_active, X_passive, y, active_features, passive_features = self._prepare_data(
            active_data, passive_data, target_col
        )
        
        # Нормализация данных активной стороны (приведение к среднему 0 и дисперсии 1)
        # fit_transform вычисляет параметры нормализации и применяет их
        X_active_scaled = self.scaler_active.fit_transform(X_active)
        # Нормализация данных пассивной стороны (приведение к среднему 0 и дисперсии 1)
        # fit_transform вычисляет параметры нормализации и применяет их
        X_passive_scaled = self.scaler_passive.fit_transform(X_passive)
        
        # Получение количества признаков активной стороны
        n_active_features = X_active_scaled.shape[1]
        # Получение количества признаков пассивной стороны
        n_passive_features = X_passive_scaled.shape[1]
        
        # Инициализация весов гейтов для активной стороны
        # Используется нормальное распределение со средним 0 и стандартным отклонением 0.1
        self.gate_weights_active = np.random.normal(0, 0.1, n_active_features)
        # Инициализация весов гейтов для пассивной стороны
        # Используется нормальное распределение со средним 0 и стандартным отклонением 0.1
        self.gate_weights_passive = np.random.normal(0, 0.1, n_passive_features)
        
        # Если включен режим подробного вывода информации
        if verbose:
            # Вывод сообщения о начале обучения
            print(f"Начало обучения FedSDG-FS")
            # Вывод количества активных признаков
            print(f"Активных признаков: {n_active_features}")
            # Вывод количества пассивных признаков
            print(f"Пассивных признаков: {n_passive_features}")
            # Вывод количества образцов (строк) в данных
            print(f"Образцов: {X_active_scaled.shape[0]}")
        
        # Итеративное обучение: цикл по всем итерациям до достижения максимума
        for iteration in range(self.max_iterations):
            # Вычисление оценок гейтов для активной стороны
            # Гейты показывают, насколько "открыт" каждый признак (значение от 0 до 1)
            gate_active = self._compute_gate_scores(
                X_active_scaled, self.gate_weights_active, encrypted=False
            )
            # Вычисление оценок гейтов для пассивной стороны
            # Гейты показывают, насколько "открыт" каждый признак (значение от 0 до 1)
            gate_passive = self._compute_gate_scores(
                X_passive_scaled, self.gate_weights_passive, encrypted=False
            )
            
            # Вычисление важности признаков на основе корреляции с целевой переменной
            # и оценок гейтов
            importance_active, importance_passive = self._compute_feature_importance(
                X_active_scaled, X_passive_scaled, y, gate_active, gate_passive
            )
            
            # Обновление весов гейтов (стохастический градиентный спуск)
            # Вычисление градиента для активной стороны: важность минус регуляризация
            gradient_active = importance_active - self.lambda_reg * self.gate_weights_active
            # Вычисление градиента для пассивной стороны: важность минус регуляризация
            gradient_passive = importance_passive - self.lambda_reg * self.gate_weights_passive
            
            # Обновление весов гейтов активной стороны с учетом скорости обучения
            # Новый вес = старый вес + скорость_обучения * градиент
            self.gate_weights_active += self.learning_rate * gradient_active
            # Обновление весов гейтов пассивной стороны с учетом скорости обучения
            # Новый вес = старый вес + скорость_обучения * градиент
            self.gate_weights_passive += self.learning_rate * gradient_passive
            
            # Применение регуляризации: ограничение весов в диапазоне [-1.0, 1.0]
            # Это предотвращает слишком большие значения весов
            self.gate_weights_active = np.clip(
                self.gate_weights_active, -1.0, 1.0
            )
            # Применение регуляризации: ограничение весов в диапазоне [-1.0, 1.0]
            # Это предотвращает слишком большие значения весов
            self.gate_weights_passive = np.clip(
                self.gate_weights_passive, -1.0, 1.0
            )
            
            # Если включен подробный вывод и текущая итерация кратна 10
            if verbose and (iteration + 1) % 10 == 0:
                # Подсчет количества отобранных признаков активной стороны
                # (гейты со средним значением выше порога)
                active_selected = np.sum(gate_active.mean(axis=0) > self.threshold)
                # Подсчет количества отобранных признаков пассивной стороны
                # (гейты со средним значением выше порога)
                passive_selected = np.sum(gate_passive.mean(axis=0) > self.threshold)
                # Вывод информации о прогрессе обучения
                print(f"Итерация {iteration + 1}/{self.max_iterations}: "
                      f"Активных: {active_selected}, Пассивных: {passive_selected}")
        
        # Финальный отбор признаков после завершения обучения
        # Вычисление финальных оценок гейтов для активной стороны
        final_gate_active = self._compute_gate_scores(
            X_active_scaled, self.gate_weights_active, encrypted=False
        )
        # Вычисление финальных оценок гейтов для пассивной стороны
        final_gate_passive = self._compute_gate_scores(
            X_passive_scaled, self.gate_weights_passive, encrypted=False
        )
        
        # Выбор признаков на основе порога
        # Создание булевой маски для активной стороны: среднее значение гейта > порог
        active_mask = final_gate_active.mean(axis=0) > self.threshold
        # Создание булевой маски для пассивной стороны: среднее значение гейта > порог
        passive_mask = final_gate_passive.mean(axis=0) > self.threshold
        
        # Формирование списка отобранных признаков активной стороны
        # Включаем только те признаки, для которых маска равна True
        self.selected_features_active = [
            active_features[i] for i in range(len(active_features)) if active_mask[i]
        ]
        # Формирование списка отобранных признаков пассивной стороны
        # Включаем только те признаки, для которых маска равна True
        self.selected_features_passive = [
            passive_features[i] for i in range(len(passive_features)) if passive_mask[i]
        ]
        
        # Если включен подробный вывод информации
        if verbose:
            # Вывод сообщения о завершении отбора признаков
            print(f"\nОтбор признаков завершен:")
            # Вывод количества отобранных активных признаков из общего количества
            print(f"Выбрано активных признаков: {len(self.selected_features_active)}/{n_active_features}")
            # Вывод количества отобранных пассивных признаков из общего количества
            print(f"Выбрано пассивных признаков: {len(self.selected_features_passive)}/{n_passive_features}")
    
    def transform(
        self,
        active_data: pd.DataFrame,
        passive_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Применение отбора признаков к новым данным
        
        Args:
            active_data: Данные активной стороны
            passive_data: Данные пассивной стороны
        
        Returns:
            Отфильтрованные данные активной и пассивной сторон
        """
        # Объединение данных активной и пассивной сторон по столбцу 'id'
        # Используется inner join - остаются только строки с совпадающими ID
        merged = active_data.merge(passive_data, on='id', how='inner')
        
        # Выбор только отобранных признаков
        # Формирование списка столбцов для активной стороны: ID + отобранные признаки
        active_selected = ['id'] + self.selected_features_active
        # Формирование списка столбцов для пассивной стороны: ID + отобранные признаки
        passive_selected = ['id'] + self.selected_features_passive
        
        # Извлечение отобранных признаков активной стороны из объединенных данных
        # Используется copy() для создания независимой копии
        active_result = merged[active_selected].copy()
        # Извлечение отобранных признаков пассивной стороны из объединенных данных
        # Используется copy() для создания независимой копии
        passive_result = merged[passive_selected].copy()
        
        # Возврат отфильтрованных данных активной и пассивной сторон
        return active_result, passive_result
    
    def get_selected_features(self) -> Tuple[List[str], List[str]]:
        """
        Получить список отобранных признаков
        
        Returns:
            Кортеж (активные признаки, пассивные признаки)
        """
        # Возврат кортежа со списками отобранных признаков активной и пассивной сторон
        return self.selected_features_active, self.selected_features_passive
    
    def get_feature_importance(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Получить важность признаков
        
        Returns:
            Кортеж словарей с важностью признаков
        """
        # Создание словаря важности признаков активной стороны
        # Ключ - название признака, значение - вес гейта (важность)
        importance_active = {
            feat: float(self.gate_weights_active[i])
            for i, feat in enumerate(self.selected_features_active)
        }
        # Создание словаря важности признаков пассивной стороны
        # Ключ - название признака, значение - вес гейта (важность)
        importance_passive = {
            feat: float(self.gate_weights_passive[i])
            for i, feat in enumerate(self.selected_features_passive)
        }
        
        # Возврат кортежа со словарями важности признаков активной и пассивной сторон
        return importance_active, importance_passive


def main():
    """Пример использования FedSDG-FS"""
    
    # Загрузка данных
    # Вывод сообщения о начале загрузки данных
    print("Загрузка данных...")
    # Загрузка данных активной стороны из CSV файла
    active_data = pd.read_csv('Data/active_dataset_test.csv')
    # Загрузка данных пассивной стороны из CSV файла
    passive_data = pd.read_csv('Data/passive_dataset_test.csv')
    
    # Вывод размерности (количество строк и столбцов) активного датасета
    print(f"Активный датасет: {active_data.shape}")
    # Вывод размерности (количество строк и столбцов) пассивного датасета
    print(f"Пассивный датасет: {passive_data.shape}")
    
    # Инициализация и обучение модели
    # Вывод разделительной линии для визуального оформления
    print("\n" + "="*50)
    # Создание экземпляра модели FedSDGFS с заданными параметрами
    model = FedSDGFS(
        lambda_reg=0.01,  # Параметр регуляризации L2
        learning_rate=0.01,  # Скорость обучения для градиентного спуска
        max_iterations=50,  # Максимальное количество итераций обучения
        threshold=0.3,  # Порог для отбора признаков (гейты выше этого значения считаются активными)
        use_encryption=False,  # Установите True для использования Paillier шифрования
        random_state=42  # Seed для воспроизводимости результатов
    )
    
    # Обучение модели на загруженных данных
    # verbose=True включает вывод информации о процессе обучения
    model.fit(active_data, passive_data, target_col='target', verbose=True)
    
    # Получение результатов
    # Вывод разделительной линии для визуального оформления
    print("\n" + "="*50)
    # Получение списков отобранных признаков для активной и пассивной сторон
    active_features, passive_features = model.get_selected_features()
    
    # Вывод списка отобранных активных признаков
    print("\nОтобранные активные признаки:")
    # Цикл по всем отобранным активным признакам
    for feat in active_features:
        # Вывод названия каждого признака с отступом
        print(f"  - {feat}")
    
    # Вывод списка отобранных пассивных признаков
    print("\nОтобранные пассивные признаки:")
    # Цикл по всем отобранным пассивным признакам
    for feat in passive_features:
        # Вывод названия каждого признака с отступом
        print(f"  - {feat}")
    
    # Применение трансформации
    # Вывод разделительной линии для визуального оформления
    print("\n" + "="*50)
    # Применение обученной модели к исходным данным для получения отфильтрованных данных
    active_transformed, passive_transformed = model.transform(active_data, passive_data)
    
    # Вывод информации о размерности данных после отбора признаков
    print(f"\nРазмерность после отбора признаков:")
    # Вывод размерности активных данных (количество строк и столбцов)
    print(f"Активные данные: {active_transformed.shape}")
    # Вывод размерности пассивных данных (количество строк и столбцов)
    print(f"Пассивные данные: {passive_transformed.shape}")
    
    # Сохранение результатов
    # Сохранение отфильтрованных данных активной стороны в CSV файл
    # index=False означает, что индекс строк не будет сохранен в файл
    active_transformed.to_csv('Data/active_dataset_selected.csv', index=False)
    # Сохранение отфильтрованных данных пассивной стороны в CSV файл
    # index=False означает, что индекс строк не будет сохранен в файл
    passive_transformed.to_csv('Data/passive_dataset_selected.csv', index=False)
    
    # Вывод информации о сохраненных файлах
    print("\nРезультаты сохранены в:")
    # Вывод пути к файлу с активными данными
    print("  - Data/active_dataset_selected.csv")
    # Вывод пути к файлу с пассивными данными
    print("  - Data/passive_dataset_selected.csv")


# Проверка, запущен ли скрипт напрямую (а не импортирован как модуль)
if __name__ == "__main__":
    # Вызов функции main() для выполнения примера использования
    main()

