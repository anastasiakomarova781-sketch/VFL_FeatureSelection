"""
Временный mock для ucbfl.ml
Только для тестирования запуска серверов
"""

class HeteroSecureBoostGuest:
    """Mock класс HeteroSecureBoostGuest"""
    def __init__(self, *args, **kwargs):
        """Принимает любые аргументы для совместимости"""
        pass
    
    @classmethod
    def from_model(cls, *args, **kwargs):
        """Mock класс-метод from_model для загрузки модели"""
        return cls()
    
    def fit(self, *args, **kwargs):
        """Mock метод fit"""
        pass
    
    def predict(self, *args, **kwargs):
        """Mock метод predict"""
        return []
    
    def get_model(self):
        """Mock метод get_model - возвращает пустой словарь"""
        return {}

class HeteroSecureBoostHost:
    """Mock класс HeteroSecureBoostHost"""
    def __init__(self, *args, **kwargs):
        """Принимает любые аргументы для совместимости"""
        pass
    
    @classmethod
    def from_model(cls, *args, **kwargs):
        """Mock класс-метод from_model для загрузки модели"""
        return cls()
    
    def fit(self, *args, **kwargs):
        """Mock метод fit"""
        pass
    
    def predict(self, *args, **kwargs):
        """Mock метод predict"""
        return []
    
    def get_model(self):
        """Mock метод get_model - возвращает пустой словарь"""
        return {}
