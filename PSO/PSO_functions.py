# Импорт библиотеки numpy для численных операций
import numpy as np

# Импорт pandas для работы с табличными данными
import pandas as pd

# Импорт классификатора дерева решений из sklearn
# Используется для оценки вклада одного признака
from sklearn.tree import DecisionTreeClassifier


class FedSDGFSPlain:
    """
    Plaintext-реализация FedSDG-FS.
    Здесь:
    - нет шифрования;
    - нет взаимодействия между сторонами;
    - сохраняется embedded-логика отбора признаков.
    """

    def __init__(self, max_depth=3, min_gain=1e-6):
        # Максимальная глубина дерева
        # В FedSDG-FS это аналог ограничения сложности SecureBoost
        self.max_depth = max_depth

        # Минимальный порог gain:
        # если вклад признака меньше — он отбрасывается
        self.min_gain = min_gain

        # Список отобранных признаков (будет заполнен после fit)
        self.selected_features_ = None

        # Словарь вида {feature_name: gain}
        self.feature_gains_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Основной метод обучения отбора признаков.

        X — матрица признаков (только активная сторона)
        y — таргет (есть только у активной стороны)
        """

        # Создаем пустой словарь для хранения gain каждого признака
        gains = {}

        # Проходим по каждому признаку по отдельности
        for feature in X.columns:
            # Берем ОДИН признак
            # Это имитирует SecureBoost, где каждый признак
            # оценивается независимо при выборе сплита
            gain = self._compute_feature_gain(X[[feature]], y)

            # Сохраняем gain признака
            gains[feature] = gain

        # Сохраняем все gains в объекте
        self.feature_gains_ = gains

        # Embedded feature selection:
        # оставляем только признаки, вклад которых больше порога
        self.selected_features_ = [
            f for f, g in gains.items() if g > self.min_gain
        ]

        # Возвращаем self для chaining
        return self

    def transform(self, X: pd.DataFrame):
        """
        Оставляет в данных только отобранные признаки
        """
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Удобный метод: fit + transform за один вызов
        """
        self.fit(X, y)
        return self.transform(X)

    def _compute_feature_gain(self, X_feat, y):
        """
        КЛЮЧЕВОЙ МЕТОД.

        Здесь вычисляется аналог information gain
        для одного признака — так же, как в SecureBoost.
        """

        # Создаем дерево решений глубины 1 (decision stump)
        # Это эквивалент одного узла SecureBoost
        tree = DecisionTreeClassifier(
            max_depth=1,
            criterion="gini"
        )

        # Обучаем дерево ТОЛЬКО по одному признаку
        tree.fit(X_feat, y)

        # impurity в корне дерева (до сплита)
        impurity_parent = tree.tree_.impurity[0]

        # Если дерево не смогло сделать сплит
        # (признак бесполезен)
        if tree.tree_.node_count == 1:
            return 0.0

        # Индексы левого и правого узлов
        left = tree.tree_.children_left[0]
        right = tree.tree_.children_right[0]

        # Количество объектов в корне
        n = tree.tree_.n_node_samples[0]

        # Количество объектов в левой и правой ветках
        n_l = tree.tree_.n_node_samples[left]
        n_r = tree.tree_.n_node_samples[right]

        # Взвешенная impurity после сплита
        impurity = (
            (n_l / n) * tree.tree_.impurity[left] +
            (n_r / n) * tree.tree_.impurity[right]
        )

        # Information gain
        gain = impurity_parent - impurity

        # Возвращаем вклад признака
        return gain

