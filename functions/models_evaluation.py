from copy import deepcopy

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


## Функция для расчета метрики модели на кросс-валидации
def cv_score(
        X,  # dataframe
        y,  # target vector
        model,  # regressor, classifier or pipeline
        metric=roc_auc_score,  # metric from competition
        n_splits=5,
        shuffle=False,
        random_state=None:

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    metrics = list()

    # Кросс-валидация
    for i, (train_index, valid_index) in enumerate(kf.split(X, y)):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index].values

        X_valid = X.loc[valid_index]
        y_valid = y.loc[valid_index].values

        model_kf = deepcopy(model)

        model_kf.fit(X_train, y_train)

        prediction = model_kf.predict_proba(X_valid)[:,1]
        cur_metric = metric(y_valid, prediction)

        metrics.append(cur_metric)

    return metrics
