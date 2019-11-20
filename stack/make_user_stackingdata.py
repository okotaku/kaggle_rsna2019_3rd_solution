import gc
import time

import feather
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from contextlib import contextmanager


dcm_feats = ['ImagePositionPatient',
       'ImageOrientationPatient', 'SamplesPerPixel', 'Rows', 'Columns', 'PixelSpacing',
       'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
       'WindowCenter', 'WindowWidth', 'RescaleIntercept', 'RescaleSlope',
       'MultiImagePositionPatient', 'ImagePositionPatient1',
       'MultiImageOrientationPatient',
       'ImageOrientationPatient1', 'ImageOrientationPatient2',
       'ImageOrientationPatient3', 'ImageOrientationPatient4',
       'ImageOrientationPatient5', 'MultiPixelSpacing', 'PixelSpacing1',
       'img_min', 'img_max', 'img_mean', 'img_std', 'img_pct_window',
       'MultiWindowCenter', 'WindowCenter1', 'MultiWindowWidth',
       'WindowWidth1']


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def preprocess_train_result(df, prefix):
    df[['ID', 'Image', 'Diagnosis']] = df['ID'].str.split('_', expand=True)
    df = df[['Image', 'Diagnosis', 'Label']]
    df.drop_duplicates(inplace=True)
    df = df.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
    df['Image'] = 'ID_' + df['Image']

    df = df.rename(columns={"any": prefix + "_any", "epidural": prefix + "_epidural",
                            "intraparenchymal": prefix + "_intraparenchymal",
                            "intraventricular": prefix + "_intraventricular", "subarachnoid": prefix + "_subarachnoid",
                            "subdural": prefix + "_subdural"})
    new_cols = [prefix + c for c in
                ["_any", "_epidural", "_intraparenchymal", "_intraventricular", "_subarachnoid", "_subdural"]]
    df = df[["Image"] + new_cols]

    return df, new_cols


with timer('load data'):
    train_label = feather.read_dataframe("../input/labels_st2.fth")
    train_meta = feather.read_dataframe("../input/df_trn.fth")
    train_label = train_label[train_label.ID != "ID_6431af929"].reset_index(drop=True)
    train_meta = train_meta[train_meta.SOPInstanceUID != "ID_6431af929"].reset_index(drop=True)
    test_meta = feather.read_dataframe("../input/df_tst.fth")
    train_meta.drop(dcm_feats, axis=1, inplace=True)
    test_meta.drop(dcm_feats, axis=1, inplace=True)

with timer('load train pred'):
    train_result = pd.read_csv("../input/exp10_seresnext_train.csv")
    train_result10 = pd.read_csv("../input/exp10_seresnext_train.csv")
    train_result16 = pd.read_csv("../input/exp16_seres_train.csv")
    train_result17 = pd.read_csv("../input/exp17_seresnext_train.csv")
    train_result18 = pd.read_csv("../input/exp18_seres_train.csv")
    train_result19 = pd.read_csv("../input/exp19_seres_train.csv")
    train_result21 = pd.read_csv("../input/exp21_seres_train.csv")
    train_result22 = pd.read_csv("../input/exp22_seres_train.csv")
    train_result23 = pd.read_csv("../input/exp23_seres_train.csv")
    train_result24 = pd.read_csv("../input/exp24_seres_train.csv")
    train_result["Label"] = train_result["Label"] * 0.025 + train_result16["Label"] * 0.2 + train_result17[
        "Label"] * 0.1 + \
                            train_result18["Label"] * 0.025 + train_result19["Label"] * 0.1 + train_result21[
                                "Label"] * 0.1 + train_result23["Label"] * 0.025 + \
                            train_result24["Label"] * 0.025 + train_result22["Label"] * 0.2
    train_result["Label"] = train_result["Label"] / (0.025 + 0.2 + 0.1 + 0.025 + 0.1 + 0.1 + 0.025 + 0.025 + 0.2)

with timer('load test pred'):
    df16 = pd.read_csv("../output/exp16_seres_sub_st2.csv")
    df10 = pd.read_csv("../output/exp10_seresnext_sub_st2.csv")
    df16_ = pd.read_csv("../output/exp16_seres_sub_st2.csv")
    df17 = pd.read_csv("../output/exp17_seresnext_sub_st2.csv")
    df18 = pd.read_csv("../output/exp18_seres_sub_st2.csv")
    df19 = pd.read_csv("../output/exp19_seres_sub_st2.csv")
    df21 = pd.read_csv("../output/exp21_seres_sub_st2.csv")
    df22 = pd.read_csv("../output/exp22_seres_sub_st2.csv")
    df23 = pd.read_csv("../output/exp23_seres_sub_st2.csv")
    df24 = pd.read_csv("../output/exp24_seres_sub_st2.csv")
    df16["Label"] = df10["Label"] * 0.025 + df16["Label"] * 0.2 + df17["Label"] * 0.1 + \
                    df18["Label"] * 0.025 + df19["Label"] * 0.1 + df21["Label"] * 0.1 + df23["Label"] * 0.025 + df24[
                        "Label"] * 0.025 + \
                    df22["Label"] * 0.2
    df16["Label"] = df16["Label"] / (0.025 + 0.2 + 0.1 + 0.025 + 0.1 + 0.1 + 0.025 + 0.025 + 0.2)

with timer('preprocess train'):
    train_result, pred_cols = preprocess_train_result(train_result, "pred")

    train_result10, pred_cols_ = preprocess_train_result(train_result10, "pred10")
    train_result = train_result.merge(train_result10, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result10
    gc.collect()

    train_result16, pred_cols_ = preprocess_train_result(train_result16, "pred16")
    train_result = train_result.merge(train_result16, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result16
    gc.collect()

    train_result17, pred_cols_ = preprocess_train_result(train_result17, "pred17")
    train_result = train_result.merge(train_result17, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result17
    gc.collect()

    train_result18, pred_cols_ = preprocess_train_result(train_result18, "pred18")
    train_result = train_result.merge(train_result18, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result18
    gc.collect()

    train_result19, pred_cols_ = preprocess_train_result(train_result19, "pred19")
    train_result = train_result.merge(train_result19, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result19
    gc.collect()

    train_result21, pred_cols_ = preprocess_train_result(train_result21, "pred21")
    train_result = train_result.merge(train_result21, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result21
    gc.collect()

    train_result22, pred_cols_ = preprocess_train_result(train_result22, "pred22")
    train_result = train_result.merge(train_result22, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result22
    gc.collect()

    train_result23, pred_cols_ = preprocess_train_result(train_result23, "pred23")
    train_result = train_result.merge(train_result23, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result23
    gc.collect()

    train_result24, pred_cols_ = preprocess_train_result(train_result24, "pred24")
    train_result = train_result.merge(train_result24, how="left", on="Image")
    pred_cols = pred_cols + pred_cols_
    del train_result24
    gc.collect()

with timer('preprocess test'):
    df16, _ = preprocess_train_result(df16, "pred")

    df10, _ = preprocess_train_result(df10, "pred10")
    df16 = df16.merge(df10, how="left", on="Image")
    del df10
    gc.collect()

    df16_, _ = preprocess_train_result(df16_, "pred16")
    df16 = df16.merge(df16_, how="left", on="Image")
    del df16_
    gc.collect()

    df17, _ = preprocess_train_result(df17, "pred17")
    df16 = df16.merge(df17, how="left", on="Image")
    del df17
    gc.collect()

    df18, _ = preprocess_train_result(df18, "pred18")
    df16 = df16.merge(df18, how="left", on="Image")
    del df18
    gc.collect()

    df19, _ = preprocess_train_result(df19, "pred19")
    df16 = df16.merge(df19, how="left", on="Image")
    del df19
    gc.collect()

    df21, _ = preprocess_train_result(df21, "pred21")
    df16 = df16.merge(df21, how="left", on="Image")
    del df21
    gc.collect()

    df22, _ = preprocess_train_result(df22, "pred22")
    df16 = df16.merge(df22, how="left", on="Image")
    del df22
    gc.collect()

    df23, _ = preprocess_train_result(df23, "pred23")
    df16 = df16.merge(df23, how="left", on="Image")
    del df23
    gc.collect()

    df24, pred_cols_ = preprocess_train_result(df24, "pred24")
    df16 = df16.merge(df24, how="left", on="Image")
    del df24
    gc.collect()

with timer('save'):
    merge_result = train_result.merge(train_meta, how="left", left_on="Image", right_on="SOPInstanceUID")
    del train_meta, train_result
    gc.collect()

    merge_label = train_label.merge(merge_result, how="left", left_on="ID", right_on="Image")
    del merge_result, train_label
    gc.collect()

    merge_test = df16.merge(test_meta, how="left", left_on="Image", right_on="SOPInstanceUID")
    del test_meta, df16
    gc.collect()

    merge_label.to_csv("../input/merge_label.csv", index=False)
    np.save("../input/pred_cols.npy", np.array(pred_cols))
    merge_test.to_csv("../input/merge_test.csv", index=False)