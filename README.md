# RSNA Intracranial Hemorrhage Detection
- This is the project for [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) hosted on Kaggle in 2019.
- It finished at [3rd place](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117223#latest-673643) in the competition.

# Experiments

|exp  |window   |input  |output  |comment   |
|:---:|:---:|:---:|:---:|:---:|
|exp7_seres_precenter_512  |subdural  |st  |st  |for external data pseudo labeling  |
|exp10_seres_threecenter_rescale_512  |3types  |st  |st  |  |
|exp16_seres_concat  |subdural  |st-1, st, st+1  |st  |  |
|exp17_7_external  |subdural  |st  |st  |with external data  |
|exp18_seres_concat_three  |3types  |st-1, st, st+1  |st  |  |
|exp19_seres_doubleconcat  |subdural  |st-2, st, st+2  |st  |  |
|exp21_seres_doublepre  |subdural  |st-2, st-1, st  |st  |  |
|exp22_seres_doublepost  |subdural  |st, st+1, st+2  |st  |  |
|exp23_seres_doublepre_three  |3types  |st-2, st-1, st  |st  |  |
|exp24_seres_doublepost_three  |3types  |st, st+1, st+2  |st  |  |
|exp25_seres_conc3  |subdural  |mean(st-3, st-2, st-1), st, mean(st+1, st+2, st+3)  |st  |  |
|exp26_seres_concall_prepost  |subdural  |mean(all st's), st, mean(st-1, st+1)  |st  |  |
|exp27_seres_conc5  |subdural  |mean(st-5, st-4, st-3, st-2, st-1), st, mean(st+1, st+2, st+3, st+4, st+5)  |st  |  |
|exp28_seres101_16  |subdural  |st-1, st, st+1  |st  |seresnext101  |
|exp32_seres_conc_any  |subdural  |st-1, st, st+1  |st  |predict 5 classes and fill_“any” on max prediction.  |
|exp34_seres_threetarget  |subdural  |st-1, st, st+1  |st-1, st, st+1  |  |
|exp36_seres_double_threetarget  |subdural  |st-2, st, st+2  |st-2, st, st+2  |  |

# Usage
1. Download [datasets](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) in input dir.

2. preprocessing
```sh
sh bin/preprocess.sh
```

3. train

```sh
sh bin/train.sh
```

4. predict
```sh
sh bin/predict.sh
```

4. stacking
sry this part of code is super dirty...

run [kernel](https://www.kaggle.com/takuok/rsna-ensemble-post-user-agg-and-basic-st2).  
then download output file in output dir.

```sh
sh bin/stack.sh
```

# Usage Demo
only predition with pretrained models.
This pipeline use user stacking only. The private score=0.04393.

1.Download models from [kaggle datasets](https://www.kaggle.com/takuok/rsna-3rdplace-models).

2.Run prediction
```sh
sh bin/demo.sh
```

# Hardware and environment
[GCP deep learning vm](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning?q=deep%20learning%20vm&id=8857b4a3-f60f-40b2-9b32-22b4428fd256&project=dena-ai-training-35-gcp&organizationId=683655960516)
- gpu: V100*2
- cpu: n1-standard-16(16 vCPU, 60GM RAM)

# Install
```sh
pip install -r requirements.txt
```

# License
The license is MIT.
