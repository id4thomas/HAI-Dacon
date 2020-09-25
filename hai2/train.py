import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#My Utils

import random

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

#Performance Metrics
from TaPR_pkg import etapr

import argparse

parser = argparse.ArgumentParser(description='HAI Train Params')
parser.add_argument('--ep', type=int, default=32,
                    help='Max Epochs')
parser.add_argument('--batch', type=int, default=512,
                    help='Batch Size')
parser.add_argument('--hid', type=int, default=100,
                    help='RNN Hidden Size')
parser.add_argument('--n_l', type=int, default=3,
                    help='Number of RNN Layers')
parser.add_argument('--win', type=int, default=60,
                    help='Window Size')
parser.add_argument('--st', type=int, default=3,
                    help='Stride')
parser.add_argument('--th', type=float, default='0.04',
                    help='Threshold')
parser.add_argument('--seed', type=int, default=42,
                    help='Random Seed')
args = parser.parse_args()

DEVICE = torch.device('cuda:0')
SEED=args.seed

#Set Random Seed
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

#Import Models
from models import *

#Get Arguments
WINDOW_SIZE = args.win
WINDOW_GIVEN = WINDOW_SIZE-1
STRIDE=args.st


#Load Train Data
TRAIN_DATASET = sorted([x for x in Path('./data/training/').glob("*.csv")])

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)

TIMESTAMP_FIELD = "time"
ATTACK_FIELD = "attack"

#Min Max Vals from Train data -> Normalize Train,Test Dataset
TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()

def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
