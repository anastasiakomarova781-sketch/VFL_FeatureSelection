"""
Временный mock для ucbfl.arch.federation.backends.rabbitmq
Только для тестирования запуска серверов
"""

class RabbitmqFederation:
    """Mock класс RabbitmqFederation"""
    def __init__(self, *args, **kwargs):
        """Принимает любые аргументы для совместимости"""
        pass
    
    @classmethod
    def from_conf(cls, *args, **kwargs):
        """Mock класс-метод from_conf"""
        return cls()
