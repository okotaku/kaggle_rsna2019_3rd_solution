import os
import random

import numpy as np
import torch


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def postprocess_multitarget(df):
    df_ = df[["ID", "any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]]
    df_ = df_.set_index("ID")
    df_ = df_.unstack().reset_index()
    df_["ID"] = df_["ID"] + "_" + df_["level_0"]
    df_ = df_.rename(columns={0: "Label"})
    df_ = df_.drop("level_0", axis=1)

    df_pre = df[
        ["PRE_ID", "pre_any", "pre_epidural", "pre_intraparenchymal", "pre_intraventricular", "pre_subarachnoid",
         "pre_subdural"]]
    df_pre = df_pre.groupby("PRE_ID").mean().reset_index()
    df_pre = df_pre.set_index("PRE_ID")
    df_pre = df_pre.unstack().reset_index()
    df_pre["PRE_ID"] = df_pre["PRE_ID"] + "_" + df_pre["level_0"].str.replace("pre_", "")
    df_pre = df_pre.rename(columns={0: "pre_Label"})
    df_pre = df_pre.drop("level_0", axis=1)

    df_post = df[
        ["POST_ID", "post_any", "post_epidural", "post_intraparenchymal", "post_intraventricular", "post_subarachnoid",
         "post_subdural"]]
    df_post = df_post.groupby("POST_ID").mean().reset_index()
    df_post = df_post.set_index("POST_ID")
    df_post = df_post.unstack().reset_index()
    df_post["POST_ID"] = df_post["POST_ID"] + "_" + df_post["level_0"].str.replace("post_", "")
    df_post = df_post.rename(columns={0: "post_Label"})
    df_post = df_post.drop("level_0", axis=1)

    df_ = df_.merge(df_pre, how="left", left_on="ID", right_on="PRE_ID")
    df_ = df_.drop("PRE_ID", axis=1)
    df_ = df_.merge(df_post, how="left", left_on="ID", right_on="POST_ID")
    df_ = df_.drop("POST_ID", axis=1)
    df_["Label"] = df_[["Label", "pre_Label", "post_Label"]].mean(1)
    df_ = df_[["ID", "Label"]]

    return df_