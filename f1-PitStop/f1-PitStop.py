import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


# Data Loading
# train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e5/train.csv")
# test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e5/test.csv")


# orig = pd.read_csv("/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv")
print("Train shape:", train.shape)
print("Test shape :", test.shape)

print(train['PitNextLap'].value_counts(normalize=True))


# print("Orig shape :", orig.shape)
train.describe()
test.describe()