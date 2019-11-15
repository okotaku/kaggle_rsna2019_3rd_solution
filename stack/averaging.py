import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df10 = pd.read_csv("../output/exp10_seresnext_sub_st2.csv")
df13 = pd.read_csv("../output/exp13_seres_sub_st2.csv")
df16 = pd.read_csv("../output/exp16_seres_sub_st2.csv")
df17 = pd.read_csv("../output/exp17_seresnext_sub_st2.csv")
df18 = pd.read_csv("../output/exp18_seres_sub_st2.csv")
df19 = pd.read_csv("../output/exp19_seres_sub_st2.csv")
df21 = pd.read_csv("../output/exp21_seres_sub_st2.csv")
df22 = pd.read_csv("../output/exp22_seres_sub_st2.csv")
df23 = pd.read_csv("../output/exp23_seres_sub_st2.csv")
df24 = pd.read_csv("../output/exp24_seres_sub_st2.csv")
df25 = pd.read_csv("../output/exp25_seres_sub_st2.csv")
df26 = pd.read_csv("../output/exp26_seres_sub_st2.csv")
df27 = pd.read_csv("../output/exp27_seres_sub_st2.csv")
df28 = pd.read_csv("../output/exp28_seres_sub_st2.csv")
df32 = pd.read_csv("../output/exp32_seres_sub_st2.csv")
df34 = pd.read_csv("../output/exp34_seres_sub_st2.csv")
df36 = pd.read_csv("../output/exp36_seres_sub_st2.csv")
app0 = pd.read_csv("../output/model001_fold0_ep1_test_tta5_st2.csv")
app1 = pd.read_csv("../output/model001_fold1_ep1_test_tta5_st2.csv")
app2 = pd.read_csv("../output/model001_fold2_ep2_test_tta5_st2.csv")
app3 = pd.read_csv("../output/model001_fold3_ep2_test_tta5_st2.csv")
app4 = pd.read_csv("../output/model001_fold4_ep2_test_tta5_st2.csv")
ens_post_double = pd.read_csv("../output/sub_st2.csv")
ens_stack = pd.read_csv("../output/sub_userstack_st2.csv")

app0 = df10[["ID"]].merge(app0, how="left", on="ID")
app1 = df10[["ID"]].merge(app1, how="left", on="ID")
app2 = df10[["ID"]].merge(app2, how="left", on="ID")
app3 = df10[["ID"]].merge(app3, how="left", on="ID")
app4 = df10[["ID"]].merge(app4, how="left", on="ID")
app0["Label"] = (app0["Label"] + app1["Label"] + app2["Label"] + app3["Label"] + app4["Label"]) / 5
df16 = df10[["ID"]].merge(df16, how="left", on="ID")
df34 = df10[["ID"]].merge(df34, how="left", on="ID")
df36 = df10[["ID"]].merge(df36, how="left", on="ID")
ens_post_double = df10[["ID"]].merge(ens_post_double, how="left", on="ID")
ens_stack = df10[["ID"]].merge(ens_stack, how="left", on="ID")

corr = pd.DataFrame({
    "df17": df17["Label"].values,
    "df10": df10["Label"].values,
    "df13": df13["Label"].values,
    "df16": df16["Label"].values,
    "df18": df18["Label"].values,
    "df19": df19["Label"].values,
    "df21": df21["Label"].values,
    "df22": df22["Label"].values,
    "df23": df23["Label"].values,
    "df24": df24["Label"].values,
    "df25": df25["Label"].values,
    "df26": df26["Label"].values,
    "df27": df27["Label"].values,
    "df28": df28["Label"].values,
    "df32": df32["Label"].values,
    "df34": df34["Label"].values,
    "df36": df36["Label"].values,
    "app0": app0["Label"].values,
    "ens_post_double": ens_post_double["Label"].values,
    "ens_stack": ens_stack["Label"].values,
})
print(corr.corr())

df16["Label"] = df16["Label"]*0.1 + df22["Label"]*0.1 + df21["Label"]*0.075 + df19["Label"]*0.075 + \
    df25["Label"]*0.05 + df28["Label"]*0.05 + df26["Label"]*0.025 + df27["Label"]*0.025 +\
    df18["Label"]*0.025 + df23["Label"]*0.025 + df24["Label"]*0.025 + \
    df17["Label"]*0.075+df10["Label"]*0.025 + df13["Label"]*0.05 + df32["Label"]*0.075 + df34["Label"]*0.1 + df36["Label"]*0.1
df16["Label"] = df16["Label"]*0.45 + ens_post_double["Label"]*0.225 + ens_stack["Label"]*0.225 + app0["Label"]*0.1
df16["Label"] = np.clip(df16["Label"], 1e-6, 1-1e-6)

df16.to_csv("ens.csv", index=False)