#!/usr/bin/env python3
"""
Скрипт для проверки содержимого обученной модели
"""
import pickle
import os
import sys

def check_model(model_path):
    """Проверяет содержимое модели"""
    if not os.path.exists(model_path):
        print(f"❌ Файл модели не найден: {model_path}")
        return
    
    print(f"📁 Проверка модели: {model_path}")
    print("=" * 60)
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Файл успешно загружен")
        print(f"📊 Тип модели: {type(model)}")
        print(f"📏 Размер файла: {os.path.getsize(model_path)} байт")
        
        if isinstance(model, dict):
            print(f"🔑 Количество ключей: {len(model)}")
            if len(model) == 0:
                print("⚠️  ВНИМАНИЕ: Модель пустая (пустой словарь {})")
                print("\n💡 Причина: Используется mock ucbfl, который не выполняет реальное обучение.")
                print("   Mock возвращает пустой словарь для тестирования инфраструктуры.")
                print("\n📝 Для реального обучения нужен настоящий ucbfl framework.")
            else:
                print("\n📋 Ключи модели:")
                for key in model.keys():
                    value = model[key]
                    if isinstance(value, (dict, list)):
                        print(f"  - {key}: {type(value).__name__} (размер: {len(value)})")
                    else:
                        print(f"  - {key}: {type(value).__name__} = {value}")
        else:
            print(f"📄 Содержимое: {model}")
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Путь к модели активной стороны
    active_model = "vfl-master@73591d69c04/example/workdir/active/models/result_model.pkl"
    
    # Путь к модели пассивной стороны
    passive_model = "vfl-master@73591d69c04/example/workdir/passive/models/result_model.pkl"
    
    print("🔍 Проверка обученных моделей VFL\n")
    
    if len(sys.argv) > 1:
        # Если указан путь как аргумент
        check_model(sys.argv[1])
    else:
        # Проверяем обе модели
        print("1️⃣ Активная сторона (Guest):")
        check_model(active_model)
        print("\n" + "=" * 60 + "\n")
        print("2️⃣ Пассивная сторона (Host):")
        check_model(passive_model)

