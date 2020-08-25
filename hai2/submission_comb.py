#HAI1 RNN Train file
import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

#ROC Curve
from perf_utils import *

import argparse

parser = argparse.ArgumentParser(description='HAI Train Params')
# parser.add_argument('--ep', type=int, default=32,
#                     help='Max Epochs')
parser.add_argument('--batch', type=int, default=512,
                    help='Batch Size')
parser.add_argument('--hid', type=int, default=100,
                    help='RNN Hidden Size')
parser.add_argument('--n_l', type=int, default=3,
                    help='Number of RNN Layers')
parser.add_argument('--win', type=int, default=90,
                    help='Window Size')
parser.add_argument('--ldim', type=int, default=8,
                    help='Latent dim')
parser.add_argument('--m_name', type=str, default='baseline',
                    help='Model Name')
parser.add_argument('--r_type', type=str, default='pca',
                    help='Reduction Algorithm')
parser.add_argument('--th', type=float, default='0.1',
                    help='Threshold')
parser.add_argument('--n_c', type=int, default=25,
                    help='Number of Components')
args = parser.parse_args()

#Set Params
WINDOW_SIZE = args.win
WINDOW_GIVEN = WINDOW_SIZE-1

N_HIDDENS = args.hid
N_LAYERS = args.n_l

L_DIM=args.ldim
# MAX_EPOCHS = args.ep
BATCH_SIZE = args.batch

MODEL_NAME=args.m_name
REDUC_TYPE=args.r_type

NUM_COMPONENTS=args.n_c
#Data Preprocess
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

TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()

#Load Recon
x_train_dist=np.load('./processed/train_{}_dist.npy'.format(REDUC_TYPE))
x_val_dist=np.load('./processed/val_{}_dist.npy'.format(REDUC_TYPE))
x_test_dist=np.load('./processed/test_{}_dist.npy'.format(REDUC_TYPE))

class HaiDataset(Dataset):
    def __init__(self, timestamps, df,reduc_dist, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.reduc_dist = np.array(reduc_dist, dtype=np.float32)
        self.valid_idxs = []
        for L in range(len(self.ts) - WINDOW_SIZE + 1):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                self.ts[L]
            ) == timedelta(seconds=WINDOW_SIZE - 1):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["r_dist"] = self.reduc_dist[i:i+WINDOW_GIVEN]
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item

#Model
if MODEL_NAME=='baseline':
    from baseline_model import HAIModel
elif MODEL_NAME=='vae':
    from vae_model import HAIModel

model_params={
    'n_tags':TRAIN_DF.shape[1],
    'n_hid': N_HIDDENS,
    'n_layers':N_LAYERS,
    'l_dim':L_DIM
}

MODEL = HAIModel(model_params)
MODEL.cuda()

######## Load #######
with open("./model/{}.pt".format(MODEL_NAME), "rb") as f:
    SAVED_MODEL = torch.load(f)

MODEL.load_state_dict(SAVED_MODEL["state"])
######## Load #######


#Load Test Data
VAL_DF = normalize(VAL_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])

HAI_DATASET_VAL = HaiDataset(
    VAL_DF_RAW[TIMESTAMP_FIELD], VAL_DF,x_val_dist, attacks=VAL_DF_RAW[ATTACK_FIELD]
)

HAI_DATASET_TEST = HaiDataset(
    TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF,x_test_dist, attacks=None
)

#Sliding Size 1 -> Check Every data
#순차적으로 보면서 예측 & 실제 차이
def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist, att = [], [], []
    reduc_dists=[]
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            if MODEL_NAME=='baseline':
                guess = model(given)
            elif MODEL_NAME=='vae':
                guess,mu,var = model(given)
            # guess = model(given)
            model_dist=torch.abs(answer - guess).cpu().numpy()
            # model_dist=torch.square(answer - guess).cpu().numpy()
            reduc_dist=batch["r_dist"].cpu().numpy()
            # reduc_dist=np.squeeze(reduc_dist, axis=2)
            # dist.append(np.concatenate([model_dist,reduc_dist],axis=1))
            dist.append(model_dist)
            # reduc_dists.append(np.squeeze(reduc_dist, axis=2))
            reduc_dists.append(reduc_dist)
            ts.append(np.array(batch["ts"]))
            # dist.append(torch.abs(answer - guess).cpu().numpy())
            try:
                att.append(np.array(batch["attack"]))
            except:
                att.append(np.zeros(batch_size))
    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(reduc_dists),
        np.concatenate(att),
    )

# MODEL.eval()
# CHECK_TS, CHECK_DIST,R_DIST, CHECK_ATT = inference(HAI_DATASET_VAL, MODEL, BATCH_SIZE)
MODEL.eval()
CHECK_TS, CHECK_DIST,R_DIST, CHECK_ATT = inference(HAI_DATASET_TEST, MODEL, BATCH_SIZE)

ANOMALY_SCORE_MODEL = np.mean(CHECK_DIST, axis=1)
ANOMALY_SCORE_REDUC = np.mean(R_DIST, axis=1)
ANOMALY_SCORE = ANOMALY_SCORE_MODEL + 3*ANOMALY_SCORE_REDUC

THRESHOLD = args.th
dist_plt=dist_graph(ANOMALY_SCORE, CHECK_ATT, piece=3,THRESHOLD=THRESHOLD)
dist_plt.savefig('./plot/dist_plot_{}.png'.format(MODEL_NAME))

dist_plt=dist_graph(ANOMALY_SCORE_MODEL, CHECK_ATT, piece=3,THRESHOLD=THRESHOLD)
dist_plt.savefig('./plot/dist_plot_{}_model.png'.format(MODEL_NAME))

dist_plt=dist_graph(ANOMALY_SCORE_REDUC, CHECK_ATT, piece=3,THRESHOLD=THRESHOLD)
dist_plt.savefig('./plot/dist_plot_{}_reduc.png'.format(MODEL_NAME))

def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs


LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)

submission = pd.read_csv('./data/sample_submission.csv')
submission.index = submission['time']
submission.loc[CHECK_TS,'attack'] = LABELS
submission.to_csv('./submission.csv', index=False)
