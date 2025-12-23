# Импорт библиотеки numpy для работы с массивами и математическими операциями
import numpy as np
# Импорт типов данных List и Dict для аннотации типов в функциях
from typing import List, Dict
# Импорт StandardScaler для нормализации данных (приведение к среднему 0 и дисперсии 1)
from sklearn.preprocessing import StandardScaler
# Импорт модуля paillier из библиотеки phe для гомоморфного шифрования
from phe import paillier

# ============================================================
# НАСТРОЙКА: Использование шифрования
# ============================================================
# Установите USE_ENCRYPTION = False для быстрого тестирования без шифрования
# Код шифрования останется в файле, но не будет использоваться
USE_ENCRYPTION = False  # False = без шифрования (быстро), True = с шифрованием (медленно, но безопасно)


# ============================================================
# Utils
# ============================================================

# Определение функции сигмоиды для преобразования значений в диапазон [0, 1]
def sigmoid(x):
    # Возврат значения сигмоидной функции: 1 / (1 + exp(-x))
    # Используется для преобразования logits в вероятности
    return 1.0 / (1.0 + np.exp(-x))


# Определение функции для сэмплирования из Concrete/Gumbel-Sigmoid распределения
def sample_concrete(logits, temperature=0.1):
    """
    Concrete / Gumbel-Sigmoid sampler
    """
    # Генерация случайных значений из равномерного распределения [0, 1]
    # Размер массива соответствует размеру logits
    u = np.random.uniform(0, 1, size=logits.shape)
    # Вычисление Gumbel шума: -log(-log(u + eps) + eps)
    # Добавление eps (1e-10) для численной стабильности
    g = -np.log(-np.log(u + 1e-10) + 1e-10)
    # Применение сигмоиды к (logits + gumbel_noise) / temperature
    # Temperature контролирует "жесткость" сэмплирования (меньше = жестче)
    return sigmoid((logits + g) / temperature)


# ============================================================
# Passive Side Logic (used by BOTH parties)
# ============================================================

# Определение класса FeatureHolder для хранения и обработки признаков
class FeatureHolder:
    """
    Логика пассивной стороны.
    Активная сторона использует ЭТО ЖЕ, но локально.
    """

    # Инициализация класса FeatureHolder
    def __init__(self, X: np.ndarray, feature_names: List[str]):
        # Нормализация признаков: приведение к среднему 0 и дисперсии 1
        # fit_transform вычисляет параметры нормализации и применяет их
        self.X = StandardScaler().fit_transform(X)
        # Сохранение количества образцов (строк) в данных
        self.n, self.d = self.X.shape
        # Сохранение списка названий признаков
        self.feature_names = feature_names

        # dual-gates (logits) - два набора logits для двойного гейта
        # logits_select - logits для отбора признаков (selection gate)
        # Инициализация случайными значениями из нормального распределения (mean=0, std=0.1)
        self.logits_select = np.random.normal(0, 0.1, self.d)
        # logits_contrib - logits для вклада признаков (contribution gate)
        # Инициализация случайными значениями из нормального распределения (mean=0, std=0.1)
        self.logits_contrib = np.random.normal(0, 0.1, self.d)

    # Метод для сэмплирования gates из Concrete распределения
    def sample_gates(self, temperature):
        # Сэмплирование selection gate из Concrete распределения
        # z определяет, будет ли признак отобран
        z = sample_concrete(self.logits_select, temperature)
        # Сэмплирование contribution gate из Concrete распределения
        # s определяет вклад признака в модель
        s = sample_concrete(self.logits_contrib, temperature)
        # Возврат произведения gates: z * s (dual-gate механизм)
        # Признак участвует только если оба gates активны
        return z * s

    # Метод для вычисления зашифрованной статистики: X^T * Enc(A)
    def compute_encrypted_statistics(self, enc_A):
        """
        Computes X^T * Enc(A)
        """
        # Проверка: используем ли мы шифрование
        if not USE_ENCRYPTION:
            # БЫСТРАЯ ВЕРСИЯ БЕЗ ШИФРОВАНИЯ (для тестирования)
            # Если enc_A - это обычная матрица (не зашифрованная), используем обычные операции
            if isinstance(enc_A, np.ndarray):
                A = enc_A
            else:
                # Преобразуем список списков в numpy массив
                A = np.array([[float(v) for v in row] for row in enc_A])
            # Вычисляем X^T * A напрямую (быстро) - матричное умножение
            return np.dot(self.X.T, A)
        
        # МЕДЛЕННАЯ ВЕРСИЯ С ШИФРОВАНИЕМ (оригинальный код - ЗАКОММЕНТИРОВАНО, но оставлено)
        # Получение количества образцов из зашифрованной матрицы A
        n, C = len(enc_A), len(enc_A[0])
        # Инициализация выходной матрицы размером (d, C) с None значениями
        # d - количество признаков, C - количество классов
        out = [[None for _ in range(C)] for _ in range(self.d)]

        # Цикл по всем признакам (d признаков)
        for j in range(self.d):
            # Цикл по всем классам (C классов)
            for c in range(C):
                # Инициализация аккумулятора для суммы
                acc = None
                # Цикл по всем образцам (n образцов)
                for i in range(n):
                    # Вычисление произведения: зашифрованное значение A[i,c] * признак X[i,j]
                    # Используется гомоморфное умножение Paillier (encrypted * scalar)
                    # КОД ШИФРОВАНИЯ (не используется если USE_ENCRYPTION = False):
                    val = enc_A[i][c] * float(self.X[i, j])  # Гомоморфное умножение Paillier
                    # Добавление к аккумулятору (первое значение или сложение)
                    # Используется гомоморфное сложение Paillier
                    # КОД ШИФРОВАНИЯ (не используется если USE_ENCRYPTION = False):
                    acc = val if acc is None else acc + val  # Гомоморфное сложение Paillier
                # Сохранение результата в выходную матрицу
                out[j][c] = acc

        # Возврат зашифрованной статистики: матрица (d, C)
        return out


# ============================================================
# Active Party
# ============================================================

# Определение класса ActiveParty для активной стороны (координатора)
class ActiveParty:
    """
    Единственная активная сторона:
    - хранит y
    - владеет ключами
    - имеет собственные признаки
    - агрегирует и обновляет
    """

    # Инициализация класса ActiveParty
    def __init__(
        self,
        X_active: np.ndarray,
        feature_names_active: List[str],
        y: np.ndarray,
        n_classes: int
    ):
        # Сохранение целевой переменной (меток) в виде целых чисел
        self.y = y.astype(int)
        # Сохранение количества образцов
        self.n = len(y)
        # Сохранение количества классов
        self.C = n_classes

        # Генерация пары ключей Paillier (публичный и приватный)
        # Публичный ключ используется для шифрования, приватный - для расшифровки
        # ЗАКОММЕНТИРОВАНО: генерация ключей (не используется если USE_ENCRYPTION = False)
        if USE_ENCRYPTION:
            self.public_key, self.private_key = paillier.generate_paillier_keypair()
        else:
            # Для режима без шифрования ключи не нужны
            self.public_key = None
            self.private_key = None

        # Активная сторона также ведет себя как пассивная (имеет свои признаки)
        # Создание локального FeatureHolder для признаков активной стороны
        self.local_holder = FeatureHolder(X_active, feature_names_active)

    # Метод для построения индикаторной матрицы классов (one-hot encoding)
    def build_indicator_matrix(self):
        # Создание матрицы нулей размером (n образцов, C классов)
        A = np.zeros((self.n, self.C))
        # Заполнение матрицы: для каждого образца устанавливаем 1 в столбце соответствующего класса
        for i, c in enumerate(self.y):
            # Установка 1 на позиции (i, c) для образца i с классом c
            A[i, c] = 1.0
        # Возврат индикаторной матрицы классов
        return A

    # Метод для шифрования индикаторной матрицы классов
    def encrypt_indicator(self, A):
        # Проверка: используем ли мы шифрование
        if not USE_ENCRYPTION:
            # БЫСТРАЯ ВЕРСИЯ БЕЗ ШИФРОВАНИЯ: возвращаем матрицу как есть
            return A.tolist() if isinstance(A, np.ndarray) else A
        
        # МЕДЛЕННАЯ ВЕРСИЯ С ШИФРОВАНИЕМ (оригинальный код)
        # Создание зашифрованной матрицы: шифрование каждого элемента матрицы A
        # Внешний список comprehension: по всем образцам (n образцов)
        # ЗАКОММЕНТИРОВАНО: шифрование каждого элемента (не используется если USE_ENCRYPTION = False)
        return [
            # Внутренний список comprehension: по всем классам (C классов)
            # Шифрование каждого элемента A[i, c] с использованием публичного ключа
            [self.public_key.encrypt(A[i, c]) for c in range(self.C)]
            for i in range(self.n)
        ]

    # Метод для расшифровки зашифрованной матрицы
    def decrypt_matrix(self, enc_mat):
        # Проверка: используем ли мы шифрование
        if not USE_ENCRYPTION:
            # БЫСТРАЯ ВЕРСИЯ БЕЗ ШИФРОВАНИЯ: возвращаем матрицу как есть
            if isinstance(enc_mat, np.ndarray):
                return enc_mat
            return np.array([[float(v) for v in row] for row in enc_mat])
        
        # МЕДЛЕННАЯ ВЕРСИЯ С ШИФРОВАНИЕМ (оригинальный код)
        # Преобразование зашифрованной матрицы в numpy массив
        # Внешний список comprehension: по всем строкам матрицы
        # ЗАКОММЕНТИРОВАНО: расшифровка каждого элемента (не используется если USE_ENCRYPTION = False)
        return np.array([
            # Внутренний список comprehension: по всем элементам строки
            # Расшифровка каждого зашифрованного значения с использованием приватного ключа
            [self.private_key.decrypt(v) for v in row]
            for row in enc_mat
        ])


# ============================================================
# FedSDG-FS (2-party, Active = Participant)
# ============================================================

# Определение класса FedSDGFS для реализации алгоритма FedSDG-FS
class FedSDGFS:
    # Инициализация класса FedSDGFS
    def __init__(
        self,
        n_classes: int,
        lr=0.05,
        lambda_reg=0.01,
        temperature=0.1,
        max_iter=50,
        threshold=0.5
    ):
        # Сохранение количества классов
        self.C = n_classes
        # Сохранение скорости обучения (learning rate) для градиентного спуска
        self.lr = lr
        # Сохранение параметра регуляризации L2 (контролирует переобучение)
        self.lambda_reg = lambda_reg
        # Сохранение температуры для Concrete/Gumbel-Sigmoid сэмплирования
        self.temperature = temperature
        # Сохранение максимального количества итераций обучения
        self.max_iter = max_iter
        # Сохранение порога для отбора признаков (gates выше порога считаются важными)
        self.threshold = threshold

    # Метод для обучения модели FedSDG-FS
    def fit(
        self,
        active: ActiveParty,
        passive: FeatureHolder
    ):
        # Активная сторона строит и шифрует индикаторную матрицу классов A
        # Построение индикаторной матрицы (one-hot encoding для меток)
        print("Построение индикаторной матрицы...")
        A = active.build_indicator_matrix()
        # Шифрование индикаторной матрицы для защиты приватности меток
        if USE_ENCRYPTION:
            print("Шифрование матрицы (это может занять время)...")
            enc_A = active.encrypt_indicator(A)
            print("Шифрование завершено. Начало итераций обучения...\n")
        else:
            print("Режим БЕЗ шифрования (быстрое тестирование). Начало итераций обучения...\n")
            enc_A = active.encrypt_indicator(A)  # Просто возвращает матрицу без шифрования

        # Итеративное обучение: цикл по всем итерациям до достижения максимума
        for iteration in range(self.max_iter):
            # Вывод прогресса каждые 5 итераций
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"Итерация {iteration + 1}/{self.max_iter}...")

            # Обе стороны (активная и пассивная) действуют как FeatureHolder
            # Создание списка holders: активная сторона (через local_holder) и пассивная сторона
            holders = [active.local_holder, passive]

            # Инициализация списков для хранения зашифрованной статистики и gates
            encrypted_stats = []
            gates = []

            # Цикл по всем holders (активная и пассивная стороны)
            for idx, h in enumerate(holders):
                # Сэмплирование gates из Concrete распределения для текущего holder
                # Gates определяют, какие признаки участвуют в текущей итерации
                g = h.sample_gates(self.temperature)
                # Сохранение gates текущего holder
                gates.append(g)
                # Вычисление зашифрованной статистики: X^T * Enc(A)
                # Статистика показывает вклад признаков в каждый класс
                # Это самая медленная часть из-за гомоморфных операций Paillier
                if (iteration + 1) % 5 == 0 or iteration == 0:
                    side_name = "активная" if idx == 0 else "пассивная"
                    print(f"  Вычисление статистики для {side_name} стороны...")
                encrypted_stats.append(h.compute_encrypted_statistics(enc_A))

            # Агрегация зашифрованной статистики на активной стороне
            if not USE_ENCRYPTION:
                # БЫСТРАЯ ВЕРСИЯ БЕЗ ШИФРОВАНИЯ: объединение статистики вертикально
                # Статистика от разных сторон имеет разную форму (разное количество признаков)
                # Поэтому объединяем их вертикально (concatenate), а не складываем
                agg = np.vstack(encrypted_stats)  # Объединение всех статистик в одну матрицу
            else:
                # МЕДЛЕННАЯ ВЕРСИЯ С ШИФРОВАНИЕМ (оригинальный код - ЗАКОММЕНТИРОВАНО, но оставлено)
                # Начинаем с статистики первого holder (активная сторона)
                agg = encrypted_stats[0]
                # Добавление статистики от остальных holders (пассивная сторона)
                for other in encrypted_stats[1:]:
                    # Цикл по всем признакам
                    for j in range(len(agg)):
                        # Цикл по всем классам
                        for c in range(len(agg[0])):
                            # Гомоморфное сложение Paillier: добавление зашифрованных значений
                            # КОД ШИФРОВАНИЯ (не используется если USE_ENCRYPTION = False):
                            agg[j][c] += other[j][c]
                # При использовании шифрования agg - это список списков, нужно преобразовать
                # Но в оригинальном коде это делается позже при расшифровке

            # Расшифровка агрегированной статистики активной стороной
            # Только активная сторона имеет приватный ключ для расшифровки
            stats = active.decrypt_matrix(agg)

            # Обновление gates для каждого holder
            # offset используется для отслеживания позиции в агрегированной статистике
            offset = 0
            # Цикл по всем holders (активная и пассивная стороны)
            for h in holders:
                # Получение количества признаков текущего holder
                d = h.d
                # Извлечение локальной статистики для текущего holder из агрегированной
                local_stats = stats[offset:offset + d]

                # Вычисление важности признаков как дисперсия по классам
                # Чем выше дисперсия, тем важнее признак для разделения классов
                importance = np.var(local_stats, axis=1)
                # Обработка NaN и бесконечных значений: заменяем на 0
                importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
                # Вычисление градиента: важность минус регуляризация
                # Регуляризация штрафует большие значения logits_select
                grad = importance - self.lambda_reg * h.logits_select
                # Обработка NaN в градиенте
                grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

                # Обновление logits_select методом градиентного спуска
                # Новый logit = старый logit + скорость_обучения * градиент
                h.logits_select += self.lr * grad
                # Обновление logits_contrib тем же градиентом
                # Оба gates обновляются одинаково для согласованности
                h.logits_contrib += self.lr * grad

                # Увеличение offset на количество признаков текущего holder
                offset += d

        # Финальный отбор признаков после завершения обучения
        # Создание словаря с отобранными признаками для активной и пассивной сторон
        self.selected = {
            # Отбор признаков для активной стороны на основе финальных logits
            "active": self._select(active.local_holder),
            # Отбор признаков для пассивной стороны на основе финальных logits
            "passive": self._select(passive)
        }

    # Вспомогательный метод для отбора признаков на основе logits
    def _select(self, holder: FeatureHolder):
        # Преобразование logits_select в вероятности с помощью сигмоиды
        # Вероятности показывают, насколько вероятно участие признака в модели
        probs = sigmoid(holder.logits_select)
        # Создание булевой маски: вероятности выше порога считаются отобранными
        mask = probs > self.threshold
        # Возврат списка названий отобранных признаков на основе маски
        # Используется list comprehension для фильтрации признаков
        return [f for f, m in zip(holder.feature_names, mask) if m]

    # Метод для получения отобранных признаков
    def get_selected_features(self):
        # Возврат словаря с отобранными признаками для активной и пассивной сторон
        return self.selected

