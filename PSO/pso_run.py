# Импорт pandas для загрузки CSV
import pandas as pd

# Импорт модуля os для работы с путями
import os

# Импорт нашего класса отбора признаков
from PSO_functions import FedSDGFSPlain


def main():
    # ===============================
    # 1. Загрузка данных
    # ===============================

    # Определяем путь к данным (работает и из корня, и из папки PSO)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Пути к датасетам
    active_data_path = os.path.join(project_root, "Data", "active_dataset_test.csv")
    passive_data_path = os.path.join(project_root, "Data", "passive_dataset_test.csv")

    # Загружаем активный датасет (с таргетом)
    df_active = pd.read_csv(active_data_path)

    # Загружаем пассивный датасет (без таргета)
    df_passive = pd.read_csv(passive_data_path)

    # Название колонки с ID
    id_col = "id"

    # Название колонки с таргетом
    target_col = "target"

    # Объединяем датасеты по id (inner join - только общие записи)
    df_merged = pd.merge(
        df_active,
        df_passive,
        on=id_col,
        how="inner"
    )

    # Сортируем по id для консистентности
    df_merged = df_merged.sort_values(by=id_col).reset_index(drop=True)

    # X — все признаки из обеих сторон (без id и target)
    X = df_merged.drop(columns=[id_col, target_col])

    # y — таргет (берем из активной стороны)
    y = df_merged[target_col]

    print(f"Загружено образцов: {len(df_merged)}")
    print(f"Активных признаков: {len([c for c in X.columns if c.startswith('feat_a_')])}")
    print(f"Пассивных признаков: {len([c for c in X.columns if c.startswith('feat_b_')])}")
    print(f"Всего признаков: {len(X.columns)}")

    # ===============================
    # 2. Отбор признаков
    # ===============================

    # Создаем объект отбора признаков
    fs = FedSDGFSPlain(
        max_depth=1,     # одно дерево = один сплит
        min_gain=1e-5    # минимальный вклад
    )

    # Обучаем и отбираем признаки
    X_selected = fs.fit_transform(X, y)

    # ===============================
    # 3. Сохранение результата
    # ===============================

    # Собираем итоговый датафрейм:
    # id + отобранные признаки + target
    result = pd.concat(
        [df_merged[[id_col]], X_selected, y],
        axis=1
    )

    # Определяем путь для сохранения (работает и из корня, и из папки PSO)
    output_path = os.path.join(project_root, "Data", "active_dataset_selected.csv")

    # Сохраняем CSV с отобранными признаками
    result.to_csv(
        output_path,
        index=False
    )

    # ===============================
    # 4. Логирование
    # ===============================

    print("\n" + "="*60)
    print("Отобранные признаки и их вклад:")
    print("="*60)

    # Разделяем признаки на активные и пассивные
    active_features = [f for f in fs.selected_features_ if f.startswith('feat_a_')]
    passive_features = [f for f in fs.selected_features_ if f.startswith('feat_b_')]

    print(f"\nОтобрано активных признаков: {len(active_features)}")
    print(f"Отобрано пассивных признаков: {len(passive_features)}")
    print(f"Всего отобрано признаков: {len(fs.selected_features_)}")

    print("\n--- Gain значений всех признаков ---")
    # Печатаем gain каждого признака
    for f, g in fs.feature_gains_.items():
        side = "АКТИВНЫЙ" if f.startswith('feat_a_') else "ПАССИВНЫЙ"
        selected = "✓" if f in fs.selected_features_ else "✗"
        print(f"{selected} [{side}] {f}: gain={g:.6f}")


# Точка входа в скрипт
if __name__ == "__main__":
    main()

