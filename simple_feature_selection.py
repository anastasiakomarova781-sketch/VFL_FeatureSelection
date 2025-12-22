#!/usr/bin/env python3
"""
Упрощенная версия FedSDG-FS для демонстрации отбора признаков
Работает без внешних зависимостей (только стандартная библиотека Python)
"""

import csv
import random
import math

def read_csv_data(filepath):
    """Чтение данных из CSV файла"""
    data = []
    headers = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                if len(row) == len(headers):
                    data.append(row)
        return headers, data
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
        return [], []

def calculate_gini_impurity(values, targets):
    """Вычисление Gini impurity для оценки важности признака"""
    if len(values) == 0:
        return 0.5
    
    # Группировка по значениям признака
    groups = {}
    for i, val in enumerate(values):
        if val not in groups:
            groups[val] = []
        groups[val].append(targets[i])
    
    gini_list = []
    for group_targets in groups.values():
        if len(group_targets) == 0:
            continue
        
        # Подсчет классов
        class_counts = {}
        for t in group_targets:
            class_counts[t] = class_counts.get(t, 0) + 1
        
        # Вычисление вероятностей классов
        probs = [count / len(group_targets) for count in class_counts.values()]
        
        # Вычисление Gini: 1 - sum(p^2)
        gini = 1 - sum(p * p for p in probs)
        gini_list.append(gini)
    
    if len(gini_list) > 0:
        avg_gini = sum(gini_list) / len(gini_list)
        # Преобразование: чем ниже Gini, тем выше важность
        return 1 - avg_gini
    return 0.5

def compute_correlation(x, y):
    """Вычисление корреляции между двумя массивами"""
    if len(x) != len(y):
        return 0.0
    
    n = len(x)
    if n == 0:
        return 0.0
    
    # Вычисление средних
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Вычисление стандартных отклонений
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n) if n > 1 else 0
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n) if n > 1 else 0
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    # Вычисление корреляции
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    correlation = covariance / (std_x * std_y)
    
    return abs(correlation) if not math.isnan(correlation) else 0.0

def select_features_simple(active_file, passive_file, threshold=0.5):
    """Упрощенный отбор признаков"""
    
    print("="*70)
    print("ОТБОР ПРИЗНАКОВ МЕТОДОМ FedSDG-FS (упрощенная версия)")
    print("="*70)
    
    # Чтение данных активного клиента
    print("\n📊 Загрузка данных активного клиента...")
    active_headers, active_data = read_csv_data(active_file)
    if not active_headers:
        return
    
    # Извлечение признаков и таргета
    feature_cols = [col for col in active_headers if col not in ['id', 'target']]
    target_idx = active_headers.index('target') if 'target' in active_headers else -1
    
    print(f"   Найдено признаков: {len(feature_cols)}")
    print(f"   Образцов: {len(active_data)}")
    
    # Подготовка данных для активного клиента
    active_features_data = {}
    targets = []
    
    for col in feature_cols:
        active_features_data[col] = []
    
    for row in active_data[:1000]:  # Ограничиваем для скорости
        if target_idx >= 0 and target_idx < len(row):
            try:
                target_val = float(row[target_idx]) if row[target_idx] else 0
                targets.append(target_val)
                
                for i, col in enumerate(feature_cols):
                    col_idx = active_headers.index(col)
                    val = float(row[col_idx]) if row[col_idx] and row[col_idx] != '' else 0.0
                    active_features_data[col].append(val)
            except:
                continue
    
    # Вычисление локальных gates для активного клиента (на основе Gini)
    print("\n🔍 Вычисление важности признаков активного клиента (метрика Gini)...")
    active_gates = {}
    for col in feature_cols:
        if len(active_features_data[col]) > 0 and len(targets) > 0:
            gini_score = calculate_gini_impurity(active_features_data[col], targets)
            active_gates[col] = gini_score
        else:
            active_gates[col] = 0.5
    
    # Чтение данных пассивного клиента
    print("\n📊 Загрузка данных пассивного клиента...")
    passive_headers, passive_data = read_csv_data(passive_file)
    if not passive_headers:
        return
    
    passive_feature_cols = [col for col in passive_headers if col != 'id']
    print(f"   Найдено признаков: {len(passive_feature_cols)}")
    print(f"   Образцов: {len(passive_data)}")
    
    # Подготовка данных для пассивного клиента
    passive_features_data = {}
    for col in passive_feature_cols:
        passive_features_data[col] = []
    
    # Вычисление скрытых предсказаний (упрощенно: среднее активных признаков)
    hidden_predictions = []
    for i in range(min(len(targets), len(active_data[:1000]))):
        pred = sum(active_features_data[col][i] * active_gates.get(col, 0.5) 
                  for col in feature_cols if i < len(active_features_data[col]))
        hidden_predictions.append(pred)
    
    for row in passive_data[:1000]:  # Ограничиваем для скорости
        for col in passive_feature_cols:
            col_idx = passive_headers.index(col)
            val = float(row[col_idx]) if col_idx < len(row) and row[col_idx] and row[col_idx] != '' else 0.0
            passive_features_data[col].append(val)
    
    # Вычисление локальных gates для пассивного клиента (на основе корреляции)
    print("\n🔍 Вычисление важности признаков пассивного клиента (корреляция)...")
    passive_gates = {}
    for col in passive_feature_cols:
        if len(passive_features_data[col]) > 0 and len(hidden_predictions) > 0:
            min_len = min(len(passive_features_data[col]), len(hidden_predictions))
            correlation = compute_correlation(
                passive_features_data[col][:min_len],
                hidden_predictions[:min_len]
            )
            passive_gates[col] = correlation
        else:
            passive_gates[col] = 0.5
    
    # Агрегация глобальных gates (усреднение)
    print("\n📈 Агрегация глобальных gates...")
    all_gates = {}
    for col in feature_cols:
        all_gates[col] = active_gates.get(col, 0.5)
    for col in passive_feature_cols:
        all_gates[col] = passive_gates.get(col, 0.5)
    
    # Отбор признаков на основе порога
    print(f"\n✅ ОТБОР ПРИЗНАКОВ (порог = {threshold})...")
    
    selected_active = [col for col in feature_cols if active_gates.get(col, 0) > threshold]
    selected_passive = [col for col in passive_feature_cols if passive_gates.get(col, 0) > threshold]
    
    print("\n" + "="*70)
    print("ОТОБРАННЫЕ АКТИВНЫЕ ПРИЗНАКИ:")
    print("="*70)
    if selected_active:
        for i, feat in enumerate(selected_active, 1):
            gate_value = active_gates.get(feat, 0)
            print(f"  {i:2d}. {feat:20s} (gate = {gate_value:.3f})")
        print(f"\nВсего отобрано: {len(selected_active)} из {len(feature_cols)} признаков")
    else:
        print("  Признаки не отобраны (все gates ниже порога)")
    
    print("\n" + "="*70)
    print("ОТОБРАННЫЕ ПАССИВНЫЕ ПРИЗНАКИ:")
    print("="*70)
    if selected_passive:
        for i, feat in enumerate(selected_passive, 1):
            gate_value = passive_gates.get(feat, 0)
            print(f"  {i:2d}. {feat:20s} (gate = {gate_value:.3f})")
        print(f"\nВсего отобрано: {len(selected_passive)} из {len(passive_feature_cols)} признаков")
    else:
        print("  Признаки не отобраны (все gates ниже порога)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    select_features_simple(
        'Data/active_dataset_test.csv',
        'Data/passive_dataset_test.csv',
        threshold=0.3  # Порог для отбора признаков
    )

