#HAI -> Numpy processed File
import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import pandas as pd

from reduce_utils import *
import os
if not os.path.exists('./processed'):
    os.makedirs('./processed')

import argparse


parser = argparse.ArgumentParser(description='Data Processor Params')
parser.add_argument('--n_c', type=int, default=25,
                    help='Number of Components')
parser.add_argument('--r_type', type=str, default='pca',
                    help='Reduction Algorithm')
args = parser.parse_args()
N_C=args.n_c
R_TYPE=args.r_type


TRAIN_DATASET = sorted([x for x in Path('./data/training/').glob("*.csv")])
VAL_DATASET = sorted([x for x in Path('./data/validation/').glob("*.csv")])
TEST_DATASET = sorted([x for x in Path('./data/testing/').glob("*.csv")])

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
VAL_DF_RAW = dataframe_from_csvs(VAL_DATASET)
TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)

print('Train raw',TRAIN_DF_RAW.shape)
print('VAL raw',VAL_DF_RAW.shape)

TIMESTAMP_FIELD = "time"
ATTACK_FIELD = "attack"

VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])

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
VAL_DF = normalize(VAL_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])#This removes time&atk
TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()

print("Normalized Train {} Val {}".format(TRAIN_DF.shape,VAL_DF.shape))
def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

train_tag_vals=np.array(TRAIN_DF, dtype=np.float32)
val_tag_vals=np.array(VAL_DF, dtype=np.float32)
test_tag_vals=np.array(TEST_DF, dtype=np.float32)

np.save('./processed/train.npy', train_tag_vals)
np.save('./processed/val.npy', val_tag_vals)

val_atk=np.array(VAL_DF_RAW[ATTACK_FIELD])
np.save('./processed/val_atk.npy',val_atk)


#Reduce
x_train_r,reduc=train_reduc(train_tag_vals,R_TYPE,n_c=N_C)
x_val_r=reduc.transform(val_tag_vals)
x_test_r=reduc.transform(test_tag_vals)
print("Reduced Data to {} with {}".format(x_train_r.shape,R_TYPE))

np.save('./processed/train_{}.npy'.format(R_TYPE), x_train_r)
np.save('./processed/val_{}.npy'.format(R_TYPE), x_val_r)

#inverse
x_train_recon=inverse(train_tag_vals,reduc,R_TYPE)
x_val_recon=inverse(val_tag_vals,reduc,R_TYPE)
x_test_recon=inverse(test_tag_vals,reduc,R_TYPE)

np.save('./processed/train_{}_rec.npy'.format(R_TYPE), x_train_recon)
np.save('./processed/val_{}_rec.npy'.format(R_TYPE), x_val_recon)

x_train_dist=np.mean(np.abs(x_train_recon-train_tag_vals),axis=1)
x_val_dist=np.mean(np.abs(x_val_recon-val_tag_vals),axis=1)
x_test_dist=np.mean(np.abs(x_test_recon-test_tag_vals),axis=1)

if R_TYPE=='srp':
    x_train_dist=np.squeeze(x_train_dist, axis=2)
    x_val_dist=np.squeeze(x_val_dist, axis=2)
    x_test_dist=np.squeeze(x_test_dist, axis=2)

np.save('./processed/train_{}l1_dist.npy'.format(R_TYPE), x_train_dist)
np.save('./processed/val_{}l1_dist.npy'.format(R_TYPE), x_val_dist)
np.save('./processed/test_{}l1_dist.npy'.format(R_TYPE), x_test_dist)

#L2 Loss
x_train_dist=np.mean(np.square(x_train_recon-train_tag_vals),axis=1)
x_val_dist=np.mean(np.square(x_val_recon-val_tag_vals),axis=1)
x_test_dist=np.mean(np.square(x_test_recon-test_tag_vals),axis=1)

if R_TYPE=='srp':
    x_train_dist=np.squeeze(x_train_dist, axis=2)
    x_val_dist=np.squeeze(x_val_dist, axis=2)
    x_test_dist=np.squeeze(x_test_dist, axis=2)
    
np.save('./processed/train_{}l2_dist.npy'.format(R_TYPE), x_train_dist)
np.save('./processed/val_{}l2_dist.npy'.format(R_TYPE), x_val_dist)
np.save('./processed/test_{}l2_dist.npy'.format(R_TYPE), x_test_dist)
