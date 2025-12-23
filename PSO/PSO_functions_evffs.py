# Импорт библиотеки numpy для численных операций и работы с массивами
import numpy as np

# Импорт pandas для работы с табличными данными
import pandas as pd

# Импорт классификатора XGBoost для построения модели машинного обучения
from xgboost import XGBClassifier

# Импорт метрики roc_auc_score для оценки качества модели
from sklearn.metrics import roc_auc_score

# =========================
# 1. Вспомогательные функции
# =========================

def decode_particle(particle, feature_dims, hyper_bounds):
    """
    Декодирование частицы:
    - первые 5 элементов: гиперпараметры XGBoost (целые)
    - остальные элементы: вероятность выбора признаков участников
    """
    # Гиперпараметры
    # Преобразуем первые 5 элементов частицы в целые числа (гиперпараметры XGBoost)
    hyper_params = [int(round(particle[i])) for i in range(5)]
    
    # Ограничиваем значения гиперпараметров заданными границами
    # np.clip обрезает значения, чтобы они не выходили за пределы [min, max]
    hyper_params = np.clip(hyper_params, hyper_bounds[:, 0], hyper_bounds[:, 1])
    
    # Признаки по участникам
    # Извлекаем вероятности выбора признаков (элементы частицы с индекса 5 и далее)
    feature_probs = particle[5:]
    
    # Создаем пустой список для хранения выбранных признаков каждого участника
    feature_selection = []
    
    # Начальная позиция для итерации по признакам
    start = 0
    
    # Проходим по каждому участнику и его количеству признаков
    for d in feature_dims:
        # Генерируем случайные числа для сравнения с вероятностями
        # np.random.rand(d) создает массив из d случайных чисел от 0 до 1
        # Сравниваем вероятности с случайными числами: если вероятность > случайное число, признак выбирается
        selected = (feature_probs[start:start+d] > np.random.rand(d)).astype(int)
        
        # Добавляем маску выбранных признаков для текущего участника
        feature_selection.append(selected)
        
        # Сдвигаем начальную позицию на количество признаков текущего участника
        start += d
    
    # Возвращаем гиперпараметры и маски выбранных признаков для всех участников
    return hyper_params, feature_selection

def evaluate_particle(hyper_params, feature_selection, active_data, passive_data, y):
    """
    Оценка частицы на активном участнике с учетом пассивных участников
    """
    # Объединяем выбранные признаки всех участников
    # Выбираем признаки активного участника по маске (где feature_selection[0] == 1)
    X_selected = active_data[:, feature_selection[0] == 1]
    
    # Проходим по каждому пассивному участнику
    for i, pdata in enumerate(passive_data):
        # Выбираем признаки текущего пассивного участника по маске
        # feature_selection[i+1] потому что индекс 0 - это активный участник
        selected_passive = pdata[:, feature_selection[i+1] == 1]
        
        # Объединяем признаки активного и пассивных участников по горизонтали (axis=1)
        # np.hstack объединяет массивы по горизонтали
        X_selected = np.hstack((X_selected, selected_passive))
    
    # Строим XGBoost с гиперпараметрами
    # Создаем объект классификатора XGBoost с заданными гиперпараметрами
    model = XGBClassifier(
        n_estimators=hyper_params[0],      # Количество деревьев в ансамбле
        max_depth=hyper_params[1],          # Максимальная глубина дерева
        min_child_weight=hyper_params[2],  # Минимальный вес в листе
        gamma=hyper_params[3],              # Минимальное уменьшение потерь для разделения
        reg_lambda=hyper_params[4],         # L2 регуляризация
        use_label_encoder=False,            # Отключаем устаревший label encoder
        eval_metric='logloss'               # Метрика для оценки (логарифмическая функция потерь)
    )
    
    # Обучаем модель на выбранных признаках и метках
    model.fit(X_selected, y)
    
    # Получаем вероятности предсказания для положительного класса (класс 1)
    # predict_proba возвращает вероятности для каждого класса, берем столбец с индексом 1
    y_pred = model.predict_proba(X_selected)[:, 1]
    
    # Вычисляем метрику AUC-ROC для оценки качества модели
    auc = roc_auc_score(y, y_pred)
    
    # Возвращаем значение AUC как оценку качества частицы
    return auc

# =========================
# 2. Инициализация роя
# =========================

def initialize_swarm(popsize, feature_dims, alpha, hyper_bounds):
    """
    FIIS: инициализация роя с учетом важности признаков
    """
    # Вычисляем общее количество признаков всех участников
    n_features = sum(feature_dims)
    
    # Создаем начальный рой частиц
    # Каждая частица имеет размерность: 5 (гиперпараметры) + n_features (признаки)
    # np.random.rand создает массив случайных чисел от 0 до 1
    swarm = np.random.rand(popsize, 5 + n_features)
    
    # Гиперпараметры: первые 5 элементов
    # Инициализируем гиперпараметры случайными целыми числами в заданных границах
    for i in range(5):
        # np.random.randint генерирует случайные целые числа в диапазоне [min, max+1)
        # size=popsize означает, что генерируем по одному значению для каждой частицы
        swarm[:, i] = np.random.randint(hyper_bounds[i, 0], hyper_bounds[i, 1]+1, size=popsize)
    
    # Признаки: вероятность выбора с учетом важности
    # Для простоты используем равномерное распределение с α
    # np.random.uniform создает случайные числа в диапазоне [0, 1)
    # (popsize, n_features) - размерность массива: количество частиц × количество признаков
    swarm[:, 5:] = np.random.uniform(0, 1, (popsize, n_features))
    
    # Возвращаем инициализированный рой частиц
    return swarm

# =========================
# 3. PSO-EVFFS алгоритм
# =========================

def PSO_EVFFS(active_data, passive_data, y, feature_dims, popsize=20, Tmax=10, w=0.5, c1=1.5, c2=1.5):
    """
    Основная функция PSO-EVFFS
    active_data: данные активного участника
    passive_data: список данных пассивных участников
    y: метки активного участника
    feature_dims: список кол-ва признаков участников
    """
    # Определяем границы для гиперпараметров XGBoost
    # Каждая строка: [минимальное значение, максимальное значение]
    # [n_estimators, max_depth, min_child_weight, gamma, reg_lambda]
    hyper_bounds = np.array([[1,6], [1,6], [5,20], [1,20], [1,20]])
    
    # Инициализируем рой частиц с учетом важности признаков
    # alpha=0.4 - параметр для учета важности признаков при инициализации
    swarm = initialize_swarm(popsize, feature_dims, alpha=0.4, hyper_bounds=hyper_bounds)
    
    # Инициализируем скорости частиц нулями
    # velocities имеет ту же размерность, что и swarm
    velocities = np.zeros_like(swarm)
    
    # Личные и глобальные лидеры
    # Pbest - лучшие позиции каждой частицы (личные рекорды)
    # Копируем начальные позиции частиц как их личные лучшие позиции
    Pbest = swarm.copy()
    
    # Pbest_scores - лучшие оценки каждой частицы
    # Создаем массив нулей для хранения оценок
    Pbest_scores = np.zeros(popsize)
    
    # Вычисляем начальные оценки для каждой частицы
    for i in range(popsize):
        # Декодируем частицу: извлекаем гиперпараметры и маски выбранных признаков
        hp, fs = decode_particle(swarm[i], feature_dims, hyper_bounds)
        
        # Оцениваем качество частицы (вычисляем AUC)
        Pbest_scores[i] = evaluate_particle(hp, fs, active_data, passive_data, y)
    
    # Находим индекс частицы с лучшей оценкой (глобальный лидер)
    Gbest_idx = np.argmax(Pbest_scores)
    
    # Сохраняем позицию глобального лидера
    Gbest = Pbest[Gbest_idx].copy()
    
    # Сохраняем оценку глобального лидера
    Gbest_score = Pbest_scores[Gbest_idx]
    
    # =========================
    # Основной цикл PSO
    # =========================
    # Выполняем Tmax итераций алгоритма PSO
    for t in range(Tmax):
        # Обновляем каждую частицу в рое
        for i in range(popsize):
            # Генерируем два случайных числа для стохастического обновления скорости
            r1, r2 = np.random.rand(2)
            
            # Обновляем скорость частицы по формуле PSO
            # w - коэффициент инерции (сохраняет текущее направление движения)
            # c1*r1*(Pbest[i]-swarm[i]) - притяжение к личному лучшему решению
            # c2*r2*(Gbest-swarm[i]) - притяжение к глобальному лучшему решению
            velocities[i] = w*velocities[i] + c1*r1*(Pbest[i]-swarm[i]) + c2*r2*(Gbest-swarm[i])
            
            # Обновляем позицию частицы: добавляем скорость к текущей позиции
            swarm[i] += velocities[i]
            
            # Декодируем обновленную частицу
            hp, fs = decode_particle(swarm[i], feature_dims, hyper_bounds)
            
            # Оцениваем качество новой позиции частицы
            score = evaluate_particle(hp, fs, active_data, passive_data, y)
            
            # Обновляем личного лидера, если новая позиция лучше
            if score > Pbest_scores[i]:
                # Сохраняем новую лучшую позицию частицы
                Pbest[i] = swarm[i].copy()
                # Сохраняем новую лучшую оценку частицы
                Pbest_scores[i] = score
        
        # Обновляем глобального лидера
        # Находим индекс частицы с лучшей оценкой после обновления всех частиц
        Gbest_idx = np.argmax(Pbest_scores)
        
        # Обновляем глобального лидера, если найдена лучшая оценка
        if Pbest_scores[Gbest_idx] > Gbest_score:
            # Сохраняем новую позицию глобального лидера
            Gbest = Pbest[Gbest_idx].copy()
            # Сохраняем новую оценку глобального лидера
            Gbest_score = Pbest_scores[Gbest_idx]
        
        # Выводим информацию о текущей итерации
        print(f"Iteration {t+1}/{Tmax}, Best AUC: {Gbest_score:.4f}")
    
    # Возвращаем лучший результат
    # Декодируем позицию глобального лидера
    hp, fs = decode_particle(Gbest, feature_dims, hyper_bounds)
    
    # Возвращаем лучшие гиперпараметры, маски выбранных признаков и лучший AUC
    return hp, fs, Gbest_score

# =========================
# 4. Пример использования
# =========================

# Устанавливаем seed для воспроизводимости результатов
np.random.seed(42)

# Генерация случайных данных для примера
# active_data: данные активного участника (100 образцов, 10 признаков)
active_data = np.random.rand(100, 10)

# passive_data: список данных пассивных участников
# Первый пассивный участник: 100 образцов, 8 признаков
# Второй пассивный участник: 100 образцов, 6 признаков
passive_data = [np.random.rand(100, 8), np.random.rand(100, 6)]

# y: метки активного участника (бинарная классификация: 0 или 1)
y = np.random.randint(0, 2, 100)

# feature_dims: список количества признаков каждого участника
# [активный участник, пассивный участник 1, пассивный участник 2]
feature_dims = [10, 8, 6]

# Запускаем алгоритм PSO-EVFFS
best_hyperparams, best_features, best_auc = PSO_EVFFS(active_data, passive_data, y, feature_dims)

# Выводим результаты
print("Best hyperparameters:", best_hyperparams)
print("Selected features per participant:", best_features)
print("Best AUC:", best_auc)


