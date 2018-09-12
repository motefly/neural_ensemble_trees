import numpy as np
import lightgbm as lgb

def TrainGBDT(data, lr, num_trees, maxleaf, verbose=True):
    """
    Fits a light_gbm to some data and returns the model.
    """
    if verbose:
        print('Fitting LightGBM...')

    train_x, valid_x, test_x, train_y, valid_y, test_y = data
    objective = "regression"
    metric = "mse"
    num_class = 1
    boost_from_average = True
    n_class = train_y.shape[1]
    if n_class == 2:
        objective = "binary"
        metric = "binary_error,binary_logloss"
        boost_from_average = False
        train_y = np.argmax(train_y, axis=1).reshape(-1,1)
        test_y = np.argmax(test_y, axis=1).reshape(-1,1)
    elif n_class > 2:
        objective = "multiclass"
        metric = "multi_error,multi_logloss"
        boost_from_average = False
        train_y = np.argmax(train_y, axis=1).reshape(-1,1)
        test_y = np.argmax(test_y, axis=1).reshape(-1,1)
        num_class = n_class
    else:
        train_y = train_y.reshape(-1,1)
        test_y = test_y.reshape(-1,1)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'num_class': num_class,
        'objective': objective,
        'metric': metric,
        'num_leaves': maxleaf,
        'min_data': 40,
        'boost_from_average': boost_from_average,
        'num_threads': 6,
        'feature_fraction': 0.8,
        'bagging_freq': 3,
        'bagging_fraction': 0.9,
        'learning_rate': lr,
    }
    lgb_train_y = train_y.reshape(-1)
    lgb_test_y = test_y.reshape(-1)
    lgb_train = lgb.Dataset(train_x, lgb_train_y, params=params)
    lgb_eval = lgb.Dataset(test_x, lgb_test_y, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_trees,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=20)
    preds = gbm.predict(train_x, raw_score=True)
    if n_class >= 2:
        if n_class == 2:
            zero_preds = np.zeros(preds.shape)
            preds = np.concatenate([zero_preds.reshape(-1,1), preds.reshape(-1,1)], axis=1)
        preds = preds.reshape(-1,n_class)
        preds = softmax(preds)
    else:
        preds = preds.reshape(-1,1)
    preds = preds.astype(np.float32)
    return gbm, preds
