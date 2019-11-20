import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from contextlib import contextmanager
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


LGBM_PARAMS = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': "binary_logloss",
    'learning_rate': 0.02,
    'num_leaves': 15,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'max_depth': 7,
    'max_bin': 255,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'min_gain_to_split': 0.02,
    'verbose': -1,
    'nthread': -1,
    'seed': 0,
}
LGBM_FIT_PARAMS = {
    'num_boost_round': 50000,
    'early_stopping_rounds': 800,
    'verbose_eval': 50000,
}


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def pred_lgbm(X_test, categorical_features, feature_name,
               fold_id, lgb_params, fit_params, model_name, score_func, calc_importances=True):
    model = lgb.Booster(model_file=('{}_fold{}.txt'.format(model_name, fold_id)))
    y_pred_test = model.predict(X_test)
    y_pred_test[y_pred_test < 0] = 0

    return y_pred_test


def calc_score(y_true, y_pred):
    return log_loss(y_true, y_pred)


def do_ensemble(test_x, feats):
    y_test = np.zeros(len(test_x))
    feature_importances = pd.DataFrame()

    for fold_id in range(5):
        y_pred_test = pred_lgbm(
            test_x,
            categorical_features=None,
            feature_name=feats,
            fold_id=fold_id,
            lgb_params=LGBM_PARAMS,
            fit_params=LGBM_FIT_PARAMS,
            model_name="model",
            score_func=calc_score,
            calc_importances=False
        )
        y_test += y_pred_test / 5
    return y_test

with timer('load data'):
    merge_test = pd.read_csv("../input/merge_test.csv")
    pred_cols = list(np.load("../input/pred_cols.npy"))

with timer('create features'):
    n_window = 20
    merge_test = merge_test.sort_values(by="ImagePositionPatient2").reset_index(drop=True)
    for i in range(1, n_window+1):
        print(i)
        merge_test = pd.concat([merge_test,
                                 merge_test.groupby("SeriesInstanceUID")[pred_cols].shift(i).add_prefix("pre{}_".format(i))], axis=1)
        merge_test = pd.concat([merge_test,
                                 merge_test.groupby("SeriesInstanceUID")[pred_cols].shift(-1*i).add_prefix("post{}_".format(i))], axis=1)

with timer('train any'):
    losses = []
    pick = [c for c in pred_cols if "_any" in c]
    feats = ["pred_any"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    test_x = merge_test[feats]
    y_test_any = do_ensemble(test_x, feats)

with timer('train epidural'):
    pick = [c for c in pred_cols if "_epidural" in c]
    feats = ["pred_epidural"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    test_x = merge_test[feats]
    y_test_epidural = do_ensemble(test_x, feats)

with timer('train intraparenchymal'):
    pick = [c for c in pred_cols if "_intraparenchymal" in c]
    feats = ["pred_intraparenchymal"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    test_x = merge_test[feats]
    y_test_intraparenchymal = do_ensemble(test_x, feats)

with timer('train intraventricular'):
    pick = [c for c in pred_cols if "_intraventricular" in c]
    feats = ["pred_intraventricular"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    test_x = merge_test[feats]
    y_test_intraventricular = do_ensemble(test_x, feats)

with timer('train subarachnoid'):
    pick = [c for c in pred_cols if "_subarachnoid" in c]
    feats = ["pred_subarachnoid"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    test_x = merge_test[feats]
    y_test_subarachnoid = do_ensemble(test_x, feats)

with timer('train subdural'):
    pick = [c for c in pred_cols if "_subdural" in c]
    feats = ["pred_subdural"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    test_x = merge_test[feats]
    y_test_subdural = do_ensemble(test_x, feats)

with timer('sub'):
    sub = pd.DataFrame({
        "any": y_test_any,
        "epidural": y_test_epidural,
        "intraparenchymal": y_test_intraparenchymal,
        "intraventricular": y_test_intraventricular,
        "subarachnoid": y_test_subarachnoid,
        "subdural": y_test_subdural,
    })
    sub["ID"] = merge_test["Image"].values
    sub = sub.set_index("ID")
    sub = sub.unstack().reset_index()
    sub["ID"] = sub["ID"] + "_" + sub["level_0"]
    sub = sub.rename(columns={0: "Label"})
    sub = sub.drop("level_0", axis=1)

print(sub.head())
sub.to_csv("../output/sub_userstack_st2.csv", index=False)