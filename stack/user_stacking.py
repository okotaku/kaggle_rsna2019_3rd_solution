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


def train_lgbm(X_train, y_train, X_valid, y_valid, X_test, categorical_features, feature_name,
               fold_id, lgb_params, fit_params, model_name, score_func, calc_importances=True):
    train = lgb.Dataset(X_train, y_train,
                        categorical_feature=categorical_features,
                        feature_name=feature_name)
    valid = lgb.Dataset(X_valid, y_valid,
                        categorical_feature=categorical_features,
                        feature_name=feature_name)

    evals_result = {}
    model = lgb.train(
        lgb_params,
        train,
        valid_sets=[valid],
        valid_names=['valid'],
        evals_result=evals_result,
        **fit_params
    )
    print('Best Iteration: {}'.format(model.best_iteration))

    y_pred_train = model.predict(X_train)
    y_pred_train[y_pred_train < 0] = 0
    train_score = score_func(y_train, y_pred_train)

    y_pred_valid = model.predict(X_valid)
    y_pred_valid[y_pred_valid < 0] = 0
    valid_score = score_func(y_valid, y_pred_valid)

    model.save_model('{}_fold{}.txt'.format(model_name, fold_id))

    if X_test is not None:
        y_pred_test = model.predict(X_test)
        y_pred_test[y_pred_test < 0] = 0
    else:
        y_pred_test = None

    if calc_importances:
        importances = pd.DataFrame()
        importances['feature'] = feature_name
        importances['gain'] = model.feature_importance(importance_type='gain')
        importances['split'] = model.feature_importance(importance_type='split')
        importances['fold'] = fold_id
    else:
        importances = None

    return y_pred_valid, y_pred_test, train_score, valid_score, importances


def calc_score(y_true, y_pred):
    return log_loss(y_true, y_pred)


def do_ensemble(x, y, test_x, feats):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(x, y)

    y_oof = np.empty(len(x), )
    y_test = np.zeros(len(test_x))
    feature_importances = pd.DataFrame()

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        x_train, y_train = x.loc[train_idx], y[train_idx]
        x_val, y_val = x.loc[val_idx], y[val_idx]

        y_pred_valid, y_pred_test, train_score, valid_score, importances = train_lgbm(
            x_train, y_train, x_val, y_val, X_test=test_x,
            categorical_features=None,
            feature_name=feats,
            fold_id=fold_id,
            lgb_params=LGBM_PARAMS,
            fit_params=LGBM_FIT_PARAMS,
            model_name="model",
            score_func=calc_score,
            calc_importances=True
        )
        print('train score={}'.format(train_score))
        print('val score={}'.format(valid_score))

        y_oof[val_idx] = y_pred_valid
        y_test += y_pred_test / 5
        feature_importances = pd.concat([feature_importances, importances], axis=0, sort=False)
    return y_oof, y_test

with timer('load data'):
    merge_label = pd.read_csv("../input/merge_label.csv")
    merge_test = pd.read_csv("../input/merge_test_st2.csv")
    pred_cols = list(np.load("../input/pred_cols.npy"))

with timer('create features'):
    n_window = 20
    merge_label = merge_label.sort_values(by="ImagePositionPatient2").reset_index(drop=True)
    merge_test = merge_test.sort_values(by="ImagePositionPatient2").reset_index(drop=True)
    for i in range(1, n_window+1):
        print(i)
        merge_label = pd.concat([merge_label,
                                 merge_label.groupby("SeriesInstanceUID")[pred_cols].shift(i).add_prefix("pre{}_".format(i))], axis=1)
        merge_label = pd.concat([merge_label,
                                 merge_label.groupby("SeriesInstanceUID")[pred_cols].shift(-1*i).add_prefix("post{}_".format(i))], axis=1)
        merge_test = pd.concat([merge_test,
                                 merge_test.groupby("SeriesInstanceUID")[pred_cols].shift(i).add_prefix("pre{}_".format(i))], axis=1)
        merge_test = pd.concat([merge_test,
                                 merge_test.groupby("SeriesInstanceUID")[pred_cols].shift(-1*i).add_prefix("post{}_".format(i))], axis=1)

with timer('train any'):
    losses = []
    pick = [c for c in pred_cols if "_any" in c]
    feats = ["pred_any"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    x = merge_label[feats]
    test_x = merge_test[feats]
    y = merge_label["any"]
    y_oof_any, y_test_any = do_ensemble(x, y, test_x, feats)
    print("loss_any", log_loss(y, merge_label["pred_any"].values), log_loss(y, y_oof_any))
    losses.append(log_loss(y, y_oof_any))
    losses.append(log_loss(y, y_oof_any))

with timer('train epidural'):
    pick = [c for c in pred_cols if "_epidural" in c]
    feats = ["pred_epidural"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    x = merge_label[feats]
    test_x = merge_test[feats]
    y = merge_label["epidural"]
    y_oof_epidural, y_test_epidural = do_ensemble(x, y, test_x, feats)
    print("loss_epidural", log_loss(y, merge_label["pred_epidural"].values), log_loss(y, y_oof_epidural))
    losses.append(log_loss(y, y_oof_epidural))

with timer('train intraparenchymal'):
    pick = [c for c in pred_cols if "_intraparenchymal" in c]
    feats = ["pred_intraparenchymal"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    x = merge_label[feats]
    test_x = merge_test[feats]
    y = merge_label["intraparenchymal"]
    y_oof_intraparenchymal, y_test_intraparenchymal = do_ensemble(x, y, test_x, feats)
    print("loss_intraparenchymal", log_loss(y, merge_label["pred_intraparenchymal"].values), log_loss(y, y_oof_intraparenchymal))
    losses.append(log_loss(y, y_oof_intraparenchymal))

with timer('train intraventricular'):
    pick = [c for c in pred_cols if "_intraventricular" in c]
    feats = ["pred_intraventricular"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    x = merge_label[feats]
    test_x = merge_test[feats]
    y = merge_label["intraventricular"]
    y_oof_intraventricular, y_test_intraventricular = do_ensemble(x, y, test_x, feats)
    print("loss_intraventricular", log_loss(y, merge_label["pred_intraventricular"].values), log_loss(y, y_oof_intraventricular))
    losses.append(log_loss(y, y_oof_intraventricular))

with timer('train subarachnoid'):
    pick = [c for c in pred_cols if "_subarachnoid" in c]
    feats = ["pred_subarachnoid"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    x = merge_label[feats]
    test_x = merge_test[feats]
    y = merge_label["subarachnoid"]
    y_oof_subarachnoid, y_test_subarachnoid = do_ensemble(x, y, test_x, feats)
    print("loss_subarachnoid", log_loss(y, merge_label["pred_subarachnoid"].values), log_loss(y, y_oof_subarachnoid))
    losses.append(log_loss(y, y_oof_subarachnoid))

with timer('train subdural'):
    pick = [c for c in pred_cols if "_subdural" in c]
    feats = ["pred_subdural"] + ["pre{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick] + ["post{}_{}".format(i, p) for i in range(1, n_window+1) for p in pick]
    #feats = feats + dcm_feats
    x = merge_label[feats]
    test_x = merge_test[feats]
    y = merge_label["subdural"]
    y_oof_subdural, y_test_subdural = do_ensemble(x, y, test_x, feats)
    print("loss_subdural", log_loss(y, merge_label["pred_subdural"].values), log_loss(y, y_oof_subdural))
    losses.append(log_loss(y, y_oof_subdural))

    print(np.mean(losses))

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
sub.to_csv("../output/sub_userstack_st2.csv", index=False)