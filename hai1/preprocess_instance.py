import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import trange

#Process Data

TRAIN_DATASET = sorted([x for x in Path('data/train-dataset/').glob("*.csv")])
TEST_DATASET = sorted([x for x in Path('data/test-dataset/').glob("*.csv")])

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)

print('Train raw',TRAIN_DF_RAW.shape)
print('Cols',TRAIN_DF_RAW.columns)

print('TEST raw',TEST_DF_RAW.shape)
print('Cols',TEST_DF_RAW.columns)


TIMESTAMP_FIELD = "time"
ATTACK_FIELD = "attack"
#DROP LABEL FIELDS
USELESS_FIELDS = ["attack_P1", "attack_P2", "attack_P3"]

print('Labels atk',TRAIN_DF_RAW[ATTACK_FIELD].unique())
print('Labels atk p1',TRAIN_DF_RAW["attack_P1"].unique())
print('Labels atk p2',TRAIN_DF_RAW["attack_P2"].unique())
print('Labels atk p3',TRAIN_DF_RAW["attack_P3"].unique())

print('Labels atk',TEST_DF_RAW[ATTACK_FIELD].unique())
print('Labels atk p1',TEST_DF_RAW["attack_P1"].unique())
print('Labels atk p2',TEST_DF_RAW["attack_P2"].unique())
print('Labels atk p3',TEST_DF_RAW["attack_P3"].unique())

#Make Atk Label - All
train_atk=np.array(TRAIN_DF_RAW[ATTACK_FIELD])
test_atk=np.array(TEST_DF_RAW[ATTACK_FIELD])
np.save('dataprocessed/train_atk.npy',train_atk)
np.save('data/processed/test_atk.npy',test_atk)

#P1 Atk
# train_p1=np.array(TRAIN_DF_RAW["attack_P1"])
# test_p1=np.array(TEST_DF_RAW["attack_P1"])
# np.save('./hai1/processed/train_p1.npy',train_p1)
# np.save('./hai1/processed/test_p1.npy',test_p1)

exit()

#VALID DATA COLUMNS - to be used for training
VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop(
    [TIMESTAMP_FIELD, ATTACK_FIELD] + USELESS_FIELDS
)

VALID_COLUMNS_IN_TEST_DATASET = TEST_DF_RAW.columns.drop(
    [TIMESTAMP_FIELD, ATTACK_FIELD] + USELESS_FIELDS
)

#MIN, MAX values taken from training df
TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()


def normalize(df):
    ndf = df.copy()
    for c in df.columns: 
        #no dif in value -> make 0
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        #Normalize other values
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

#ewm: exponential weighted function - noise smoothing 
TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TEST_DATASET]).ewm(alpha=0.9).mean()

#Check if value >1 or <0 or NaN (Due to using train_df min,max vals)
def boundary_check(df):
    x = np.array(df, dtype=np.float32) 
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

print(TRAIN_DF.head)

#Sliding Window?
train_timestamps=np.array(TRAIN_DF_RAW[TIMESTAMP_FIELD])
train_tag_vals=np.array(TRAIN_DF, dtype=np.float32)

test_timestamps=np.array(TEST_DF_RAW[TIMESTAMP_FIELD])
test_tag_vals=np.array(TEST_DF, dtype=np.float32)

np.save('data/processed/train.npy', train_tag_vals)
np.save('data/processed/test.npy', test_tag_vals)

print(train_tag_vals.shape)
print(train_tag_vals[:5])

# valid_idxs=[]

# #SEE 89 WINDOW -> Predict 90th frame
# WINDOW_GIVEN = 89
# WINDOW_SIZE = 90

# stride=1

# for L in range(len(timestamps) - WINDOW_SIZE + 1):
#     R = L + WINDOW_SIZE - 1
#     # print(timestamps[R])
#     if dateutil.parser.parse(timestamps[R]) - dateutil.parser.parse(
#         timestamps[L]
#     ) == timedelta(seconds=WINDOW_SIZE - 1):
#         valid_idxs.append(L)
# valid_idxs = np.array(valid_idxs, dtype=np.int32)[::stride]
# n_idxs = len(valid_idxs)

# print(valid_idxs)