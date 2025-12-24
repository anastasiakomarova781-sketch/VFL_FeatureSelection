"""
Временный mock для ucbfl.arch.dataframe.PandasReader
Только для тестирования запуска серверов
"""
import pandas as pd

class PandasReader:
    """Mock класс PandasReader"""
    def __init__(self, *args, **kwargs):
        """Принимает любые аргументы для совместимости"""
        self.sample_id_name = kwargs.get('sample_id_name', 'sample_id')
        self.match_id_name = kwargs.get('match_id_name', 'id')
        self.dtype = kwargs.get('dtype', 'float32')
        self.label_name = kwargs.get('label_name', None)
    
    def read(self, *args, **kwargs):
        """Mock метод read"""
        return pd.DataFrame()
    
    def to_frame(self, ctx, df):
        """Mock метод to_frame"""
        return df

class DataFrame:
    """Mock класс DataFrame"""
    pass
