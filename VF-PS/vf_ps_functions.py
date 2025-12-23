# Импорт необходимых библиотек
import pandas as pd  # Для работы с таблицами и CSV-файлами
import numpy as np   # Для математических операций и работы с массивами
from sklearn.metrics import mutual_info_score  # Для вычисления взаимной информации (MI)

# -----------------------------
# 1. Загрузка данных
# -----------------------------
# Загружаем активный датасет (с признаками A1, A2, A3 и целевой переменной Y)
active_df = pd.read_csv("../Data/active_dataset_test.csv")

# Загружаем пассивный датасет (с признаками P1, P2, без таргета)
passive_df = pd.read_csv("../Data/passive_dataset_test.csv")

# Сохраняем целевую переменную Y из активного датасета
Y = active_df['target']

# -----------------------------
# 2. Вычисление MI для каждого признака
# -----------------------------
# Функция для вычисления взаимной информации между признаком и таргетом
def compute_mi(feature_series, target):
    # Удаляем строки с NaN значениями
    mask = ~(pd.isna(feature_series) | pd.isna(target))
    feature_clean = feature_series[mask]
    target_clean = target[mask]
    # Если после очистки данных не осталось, возвращаем 0
    if len(feature_clean) == 0:
        return 0.0
    return mutual_info_score(feature_clean, target_clean)

# Список активных признаков (берем первые несколько для примера)
active_features = ['feat_a_00', 'feat_a_01', 'feat_a_02']

# Вычисляем MI каждого активного признака с Y
# Результат сохраняется в словаре {имя_признака: значение_MI}
active_mi = {f: compute_mi(active_df[f], Y) for f in active_features}

# Пассивные признаки (у нас нет прямого доступа к Y)
passive_features = ['feat_b_00', 'feat_b_01']
# Для примера задаём условные значения MI (обычно вычисляется через SecureBoost/VF-MINE)
passive_mi = {'feat_b_00': 0.5, 'feat_b_01': 0.3}

# -----------------------------
# 3. Усреднение MI по участникам
# -----------------------------
# Среднее MI для активного участника
avg_mi_active = np.mean(list(active_mi.values()))

# Среднее MI для пассивного участника
avg_mi_passive = np.mean(list(passive_mi.values()))

# Словарь с оценками "ценности" каждого участника
participant_scores = {'A': avg_mi_active, 'P': avg_mi_passive}

# -----------------------------
# 4. Выбор топ-L участников (по MI)
# -----------------------------
L = 1  # Количество участников, которых хотим выбрать
# Сортируем участников по среднему MI по убыванию и выбираем топ-L
top_participants = sorted(participant_scores.items(), key=lambda x: x[1], reverse=True)[:L]

# -----------------------------
# 5. Выбор признаков внутри участников
# -----------------------------
# Словарь для хранения выбранных признаков по каждому участнику
selected_features = {}

# Проходим по выбранным топ-участникам
for p, _ in top_participants:
    if p == 'A':
        # Сортируем активные признаки по убыванию MI
        sorted_feats = sorted(active_mi.items(), key=lambda x: x[1], reverse=True)
    else:
        # Сортируем пассивные признаки по убыванию MI
        sorted_feats = sorted(passive_mi.items(), key=lambda x: x[1], reverse=True)
    
    # Сохраняем признаки с MI > 0 как выбранные
    selected_features[p] = [f for f, mi in sorted_feats if mi > 0]

# -----------------------------
# 6. Вывод результатов
# -----------------------------
# Печатаем средние MI участников
print("Средние MI участников:", participant_scores)

# Печатаем список выбранных участников
print("Выбранные участники:", [p for p, _ in top_participants])

# Печатаем выбранные признаки для каждого участника
print("Выбранные признаки по участникам:", selected_features)

