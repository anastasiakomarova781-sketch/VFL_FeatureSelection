# ============================================================
# Простое выполнение FedSDG-FS на реальных VFL датасетах
# ============================================================
# Шифрование Paillier отключено для быстрого тестирования
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# НАСТРОЙКИ ДЛЯ УСКОРЕНИЯ
# ============================================================
USE_SAMPLE = True      # True = использовать подвыборку для быстрого тестирования
SAMPLE_SIZE = 1000     # Количество образцов для тестирования
MAX_ITER = 10          # Количество итераций для теста

# ============================================================
# Импорт алгоритма и сущностей (шаг шифрования отключен)
# ============================================================
from fedsdg_fs_article import (
    ActiveParty,
    FeatureHolder,
    FedSDGFS
)

# ============================================================
# ЭТАП 1. Загрузка данных
# ============================================================
print("Загрузка данных...")
active_df = pd.read_csv("../Data/active_dataset_test.csv")
passive_df = pd.read_csv("../Data/passive_dataset_test.csv")
print(f"  Активный: {active_df.shape}, Пассивный: {passive_df.shape}")

if USE_SAMPLE:
    active_df = active_df.head(SAMPLE_SIZE)
    passive_df = passive_df.head(SAMPLE_SIZE)
    print(f"  Используем подвыборку: {SAMPLE_SIZE} образцов")

# Проверка столбцов
assert "id" in active_df.columns and "target" in active_df.columns
assert "id" in passive_df.columns

# ============================================================
# ЭТАП 2. Выравнивание объектов по id
# ============================================================
common_ids = np.intersect1d(active_df["id"].values, passive_df["id"].values)
active_df = active_df[active_df["id"].isin(common_ids)].sort_values("id")
passive_df = passive_df[passive_df["id"].isin(common_ids)].sort_values("id")
print(f"Общее количество образцов: {len(common_ids)}")

# ============================================================
# ЭТАП 3. Формирование данных для сторон
# ============================================================
y = active_df["target"].values
X_active = active_df.drop(columns=["id", "target"]).values
X_passive = passive_df.drop(columns=["id"]).values
active_features = list(active_df.drop(columns=["id", "target"]).columns)
passive_features = list(passive_df.drop(columns=["id"]).columns)
n_classes = len(np.unique(y))

print(f"Классов: {n_classes}, Активных признаков: {len(active_features)}, Пассивных признаков: {len(passive_features)}")

# ============================================================
# ЭТАП 4. Инициализация сторон
# ============================================================
active_party = ActiveParty(
    X_active=X_active,
    feature_names_active=active_features,
    y=y,
    n_classes=n_classes
)

passive_party = FeatureHolder(
    X=X_passive,
    feature_names=passive_features
)

# ============================================================
# ЭТАП 5. Инициализация модели FedSDG-FS
# ============================================================
model = FedSDGFS(
    n_classes=n_classes,
    lr=0.05,
    lambda_reg=0.01,
    temperature=0.1,
    max_iter=MAX_ITER,
    threshold=0.5
)
print(f"Итераций: {MAX_ITER}, LR: 0.05, Регуляризация: 0.01")

# ============================================================
# ЭТАП 6. Запуск обучения
# ============================================================
print("\nНачало обучения FedSDG-FS...")
model.fit(
    active=active_party,
    passive=passive_party
)
print("Обучение завершено!")

# ============================================================
# ЭТАП 7. Получение результатов
# ============================================================
selected = model.get_selected_features()

# Диагностика: проверяем значения gates для активной стороны
print("\n===== Диагностика gates =====")
from fedsdg_fs_article import sigmoid
active_probs = sigmoid(active_party.local_holder.logits_select)
print("Вероятности (gates) активных признаков:")
for i, (feat, prob) in enumerate(zip(active_features, active_probs)):
    selected_mark = "✓" if prob > 0.5 else "✗"
    print(f"  {selected_mark} {feat}: {prob:.4f}")

passive_probs = sigmoid(passive_party.logits_select)
print("\nВероятности (gates) пассивных признаков:")
for i, (feat, prob) in enumerate(zip(passive_features, passive_probs)):
    selected_mark = "✓" if prob > 0.5 else "✗"
    print(f"  {selected_mark} {feat}: {prob:.4f}")

print("\n===== Результаты отбора признаков =====")
print(f"Порог отбора: 0.5")
print(f"Активная сторона (отобрано {len(selected['active'])} из {len(active_features)}):")
if selected["active"]:
    for f in selected["active"]:
        print(f" - {f}")
else:
    print("  (признаки не отобраны - все gates ниже порога 0.5)")
print(f"\nПассивная сторона (отобрано {len(selected['passive'])} из {len(passive_features)}):")
if selected["passive"]:
    for f in selected["passive"]:
        print(f" - {f}")
else:
    print("  (признаки не отобраны - все gates ниже порога 0.5)")

# ============================================================
# ЭТАП 8. Сохранение отфильтрованных датасетов
# ============================================================
active_selected_df = active_df[["id", "target"] + selected["active"]]
active_selected_df.to_csv("../Data/active_dataset_selected.csv", index=False)

passive_selected_df = passive_df[["id"] + selected["passive"]]
passive_selected_df.to_csv("../Data/passive_dataset_selected.csv", index=False)

print("\nСохранено:")
print(" - ../Data/active_dataset_selected.csv")
print(" - ../Data/passive_dataset_selected.csv")
