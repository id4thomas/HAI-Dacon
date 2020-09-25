import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#My Utils
from utils.data_utils import *
from utils.inference_utils import *
from utils.perf_utils import *

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

#Model Params
parser.add_argument('--hid', type=int, default=100,
                    help='RNN Hidden Size')
parser.add_argument('--n_l', type=int, default=3,
                    help='Number of RNN Layers')
parser.add_argument('--do', type=float, default=0,
                    help='Dropout Ratio')

#Dataset Params
parser.add_argument('--win', type=int, default=60,
                    help='Window Size')
parser.add_argument('--st', type=int, default=3,
                    help='Stride')

#Inference params
parser.add_argument('--th', type=float, default=0.04,
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

TRAIN_DATASET = sorted([x for x in Path('./data/training/').glob("*.csv")])
VAL_DATASET = sorted([x for x in Path('./data/validation/').glob("*.csv")])
TEST_DATASET = sorted([x for x in Path('./data/testing/').glob("*.csv")])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
VAL_DF_RAW = dataframe_from_csvs(VAL_DATASET)
TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)

TIMESTAMP_FIELD = "time"
ATTACK_FIELD = "attack"

VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])

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
VAL_DF = normalize(VAL_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])
TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()

HAI_DATASET_VAL = HaiDataset(
    VAL_DF_RAW[TIMESTAMP_FIELD], VAL_DF, attacks=VAL_DF_RAW[ATTACK_FIELD]
)
HAI_DATASET_TEST = HaiDataset(
    TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF, attacks=None
)

#Load Model

gru_params={
    'n_tags':TRAIN_DF.shape[1],
    'n_hid': args.hid,
    'n_layers':args.n_l,
    'do':args.do,
}
MODEL = GRUModel(gru_params)
MODEL.cuda()
MODEL.train()

gru_name='_best'

with open("./model/gru{}.pt".format(gru_name), "rb") as f:
    BEST_MODEL = torch.load(f)

MODEL.load_state_dict(BEST_MODEL["state"])

######## EVAL #########
MODEL.eval()

THRESHOLD=args.th

#Validation
dataloader = DataLoader(HAI_DATASET_VAL, batch_size=BATCH_SIZE)
CHECK_TS, GRU_DIST , CHECK_ATT = inference(HAI_DATASET_VAL, MODEL, BATCH_SIZE)
ANOMALY_SCORE = np.mean(GRU_DIST, axis=1)

print("GRU AUC")
auc,_,_=make_roc(ANOMALY_SCORE,CHECK_ATT,ans_label=1)

LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
ATTACK_LABELS = put_labels(np.array(VAL_DF_RAW[ATTACK_FIELD]),0.5)

dist_plt=dist_graph(ANOMALY_SCORE, CHECK_ATT, piece=3,THRESHOLD=THRESHOLD)
dist_plt.savefig('./plot/val_dist.png')

FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VAL_DF_RAW[TIMESTAMP_FIELD]))
tapr_score(FINAL_LABELS,ATTACK_LABELS)


#Create Submission
MODEL.eval()

CHECK_TS, GRU_DIST, CHECK_ATT = inference(HAI_DATASET_TEST,  MODEL, BATCH_SIZE)
ANOMALY_SCORE = np.mean(GRU_DIST, axis=1)

LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)

dist_plt=dist_graph(ANOMALY_SCORE, CHECK_ATT, piece=3,THRESHOLD=THRESHOLD)
dist_plt.savefig('./plot/test_dist.png')

dist_plt=dist_graph(ANOMALY_SCORE, LABELS, piece=3,THRESHOLD=THRESHOLD)
dist_plt.savefig('./plot/test_ans.png')

submission = pd.read_csv('./data/sample_submission.csv')
submission.index = submission['time']
submission.loc[CHECK_TS,'attack'] = LABELS
submission.to_csv('./submission.csv', index=False)
