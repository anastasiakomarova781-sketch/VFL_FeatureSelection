"""
Временный mock для ucbfl.arch.config
Только для тестирования запуска серверов
"""

class Config:
    """Mock класс Config"""
    def __init__(self):
        """Инициализация mock конфига"""
        pass
    
    def temp_override(self, *args, **kwargs):
        """Mock метод temp_override - контекстный менеджер для временного переопределения конфига"""
        return self
    
    def __enter__(self):
        """Вход в контекстный менеджер"""
        return self
    
    def __exit__(self, *args, **kwargs):
        """Выход из контекстного менеджера"""
        pass

cfg = Config()
