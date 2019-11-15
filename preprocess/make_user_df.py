import feather
import pandas as pd


train_label = feather.read_dataframe("../input/labels_st2.fth")
train_meta = feather.read_dataframe("../input/df_trn.fth")
train_meta = train_meta.append(feather.read_dataframe("../input/df_tst.fth")).reset_index(drop=True)
train_meta = train_meta[["SOPInstanceUID", "PatientID", "SeriesInstanceUID", "ImagePositionPatient1", "ImagePositionPatient2"]]
train_meta["SeriesInstanceUID_"] = train_meta["SeriesInstanceUID"]
train_meta = train_meta.sort_values(by="ImagePositionPatient2").reset_index(drop=True)

for i in range(1, 6):
    gp = train_meta.groupby(["PatientID", "SeriesInstanceUID_"])[["SOPInstanceUID", "SeriesInstanceUID", "ImagePositionPatient1", "ImagePositionPatient2"]].shift(i)
    train_meta = pd.concat([train_meta, gp.rename(columns={"SOPInstanceUID": "pre{}_SOPInstanceUID".format(i),
                                                  "SeriesInstanceUID": "pre{}_SeriesInstanceUID".format(i),
                                                  "ImagePositionPatient1": "pre{}_ImagePositionPatient1".format(i),
                                                  "ImagePositionPatient2": "pre{}_ImagePositionPatient2".format(i)})], axis=1)
    gp = train_meta.groupby(["PatientID", "SeriesInstanceUID_"])[
        ["SOPInstanceUID", "SeriesInstanceUID", "ImagePositionPatient1", "ImagePositionPatient2"]].shift(-1*i)
    train_meta = pd.concat([train_meta, gp.rename(columns={"SOPInstanceUID": "post{}_SOPInstanceUID".format(i),
                                                           "SeriesInstanceUID": "post{}_SeriesInstanceUID".format(i),
                                                           "ImagePositionPatient1": "post{}_ImagePositionPatient1".format(
                                                               i),
                                                           "ImagePositionPatient2": "post{}_ImagePositionPatient2".format(
                                                               i)})], axis=1)

gp = train_meta.groupby(["PatientID", "SeriesInstanceUID"])["SOPInstanceUID"].apply(lambda x: list(x)).reset_index()
gp["SOPInstanceUID"] = gp["SOPInstanceUID"].map(lambda x: ",".join(x))
train_meta = train_meta.merge(gp.rename(columns={"SOPInstanceUID": "all_SOPInstanceUID"}), how="left", on=["PatientID", "SeriesInstanceUID"])

train = train_label.merge(train_meta[["SOPInstanceUID", "pre1_SOPInstanceUID", "post1_SOPInstanceUID", "pre2_SOPInstanceUID", "post2_SOPInstanceUID",
                                      "pre3_SOPInstanceUID", "post3_SOPInstanceUID", "pre4_SOPInstanceUID", "post4_SOPInstanceUID",
                                      "pre5_SOPInstanceUID", "post5_SOPInstanceUID", "all_SOPInstanceUID"]], how="left", left_on="ID", right_on="SOPInstanceUID")
train.rename(columns={"ID": "Image"}).to_csv("train_concat.csv", index=False)