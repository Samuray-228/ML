import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    i = np.argsort(feature_vector)
    sorted_features = feature_vector[i]
    sorted_targets = target_vector[i]
    
    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    
    left_p_0 = np.cumsum(sorted_targets[:-1] == 0) / np.arange(1, len(target_vector))
    left_p_1 = np.cumsum(sorted_targets[:-1] == 1) / np.arange(1, len(target_vector))
    right_p_0 = np.flip(np.cumsum(np.flip(sorted_targets[1:], axis=-1) == 0), axis=-1) / np.arange(len(target_vector) - 1, 0, -1)
    right_p_1 = np.flip(np.cumsum(np.flip(sorted_targets[1:], axis=-1) == 1), axis=-1) / np.arange(len(target_vector) - 1, 0, -1)
    
    H_left = 1 - left_p_0**2 - left_p_1**2
    H_right = 1 - right_p_0**2 - right_p_1**2
    
    ginis = -(np.arange(1, len(target_vector)) * H_left / len(target_vector) + np.arange(len(target_vector) - 1, 0, -1) * H_right / len(target_vector))
    
    return thresholds, ginis, thresholds[np.argmax(ginis)], np.max(ginis)


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count if current_count else 0
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError
            
            if len(np.unique(feature_vector)) < 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        if np.sum(split) >= self._min_samples_leaf and np.sum(np.logical_not(split)) >= self._min_samples_leaf:
            self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
            self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)
        else:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        if self._feature_types[node['feature_split']] == 'real':
            if x[node['feature_split']] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            if x[node['feature_split']] in node['categories_split']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

class LinearRegressionTree(DecisionTree):
    def __init__(self, feature_types, base_model_type=None, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        super().__init__(feature_types, max_depth, min_samples_split, min_samples_leaf)

    def _find_best_linear_regression_split(self, feature_vector, target_vector, quantiles=10):
        unique_values = np.sort(np.unique(feature_vector))
        thresholds = np.quantile(unique_values, q=np.linspace(0, 1, num=quantiles))

        best_threshold, best_loss, best_lr_left, best_lr_right = None, float('inf'), None, None

        for threshold in thresholds:
            split = feature_vector < threshold
            if np.sum(split) == 0 or np.sum(np.logical_not(split)) == 0:
                continue

            lr_left = LinearRegression().fit(feature_vector[split].reshape(-1, 1), target_vector[split])
            lr_right = LinearRegression().fit(feature_vector[np.logical_not(split)].reshape(-1, 1), target_vector[np.logical_not(split)])

            loss_left = np.mean((lr_left.predict(feature_vector[split].reshape(-1, 1)) - target_vector[split])**2)
            loss_right = np.mean((lr_right.predict(feature_vector[np.logical_not(split)].reshape(-1, 1)) - target_vector[np.logical_not(split)])**2)

            weighted_loss = (np.sum(split) / len(feature_vector)) * loss_left + (np.sum(np.logical_not(split)) / len(feature_vector)) * loss_right

            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_threshold = threshold
                best_lr_left = lr_left
                best_lr_right = lr_right

        return best_threshold, best_loss, best_lr_left, best_lr_right

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        feature_best, threshold_best, _, _ = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_vector = sub_X[:, feature]

            if len(np.unique(feature_vector)) < 3:
                continue

            threshold, loss, _, _ = self._find_best_linear_regression_split(feature_vector, sub_y)

            if feature_best is None or loss < threshold_best:
                feature_best = feature
                threshold_best = threshold

        if feature_best is None:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {}, {}

        split = sub_X[:, feature_best] < threshold_best
        if np.sum(split) >= self._min_samples_leaf and np.sum(np.logical_not(split)) >= self._min_samples_leaf:
            self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
            self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)
        else:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['model'].predict(x.reshape(1, -1))[0]
        if self._feature_types[node['feature_split']] == 'real':
            if x[node['feature_split']] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            raise ValueError
        
    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
