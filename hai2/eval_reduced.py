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

from data_utils import *
from reduce_utils import *

import os
if not os.path.exists('./plot'):
    os.makedirs('./plot')

parser = argparse.ArgumentParser(description='HAI Eval Params')
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

#Data Preprocess
TRAIN_DATASET = sorted([x for x in Path('./data/training/').glob("*.csv")])
VAL_DATASET = sorted([x for x in Path('./data/validation/').glob("*.csv")])
# TEST_DATASET = sorted([x for x in Path('../hai1/data/test-dataset/').glob("*.csv")])

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
VAL_DF_RAW = dataframe_from_csvs(VAL_DATASET)
# TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)

TIMESTAMP_FIELD = "time"
ATTACK_FIELD = "attack"

x_train_r=load_processed('train_'+REDUC_TYPE,only_data=True)
_,y_val=load_processed('val')
x_val_r=load_processed('val_'+REDUC_TYPE,only_data=True)

#Reduce Data Dimensions
NUM_COMPONENTS=25

class HaiDataset(Dataset):
    def __init__(self, timestamps, vals, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(vals, dtype=np.float32)
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
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item

#Model
if MODEL_NAME=='baseline':
    from baseline_model import HAIModel
elif MODEL_NAME=='vae':
    from vae_model import HAIModel

model_params={
    'n_tags':NUM_COMPONENTS,
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

HAI_DATASET_VAL = HaiDataset(
    VAL_DF_RAW[TIMESTAMP_FIELD], x_val_r, attacks=y_val
)

MODEL.eval()
dataloader = DataLoader(HAI_DATASET_VAL, batch_size=BATCH_SIZE)
ts, dist, att = [], [], []
hids=[]
mus,vars = [],[]
#Set hook to get rnn output
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output#.detach()
    return hook
MODEL.rnn.register_forward_hook(get_activation('rnn'))

print('Make Validation Hids')
with torch.no_grad():
    for batch in dataloader:
        given = batch["given"].cuda()
        answer = batch["answer"].cuda()
        if MODEL_NAME=='baseline':
            guess = MODEL(given)
        elif MODEL_NAME=='vae':
            guess,mu,var = MODEL(given)
            mus.append(mu.cpu().numpy())
            vars.append(var.cpu().numpy())
        #Get rnn vals
        outs, _ =activation['rnn']
        hid=outs[-1].cpu().numpy()
        # print(hid.shape)
        ts.append(np.array(batch["ts"]))
        dist.append(torch.abs(answer - guess).cpu().numpy())
        att.append(np.array(batch["attack"]))
        hids.append(hid)

hids=np.concatenate(hids,axis=0)
att=np.concatenate(att,axis=0)
dist=np.concatenate(dist,axis=0)

if MODEL_NAME=='vae':
    mus=np.concatenate(mus,axis=0)
    vars=np.concatenate(vars,axis=0)
    np.save('./latents/test_mu.npy',mus)
    np.save('./latents/test_var.npy',vars)
    np.save('./latents/test_atk.npy',att)


#ROC Curve
roc_fig,auc,desc=make_roc(np.mean(dist,axis=1),att,ans_label=1,make_desc=True)
roc_fig.savefig('./plot/roc_{}.png'.format(MODEL_NAME))
desc.to_csv('./desc/baseline_grp_25.csv')
plt.clf()


#Sliding Size 1 -> Check Every data
#순차적으로 보면서 예측 & 실제 차이
def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist, att = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            if MODEL_NAME=='baseline':
                guess = model(given)
            elif MODEL_NAME=='vae':
                guess,mu,var = model(given)
            # guess = model(given)
            ts.append(np.array(batch["ts"]))
            dist.append(torch.abs(answer - guess).cpu().numpy())
            att.append(np.array(batch["attack"]))
    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(att),
    )

MODEL.eval()
CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_VAL, MODEL, BATCH_SIZE)

ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

#Plot graph of recon loss at every point
dist_plt=dist_graph(ANOMALY_SCORE, CHECK_ATT, piece=3)
dist_plt.savefig('./plot/dist_plot_{}.png'.format(MODEL_NAME))

def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

THRESHOLD = 0.098
LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)

#Make Prediction Label
ATTACK_LABELS = put_labels(y_val,THRESHOLD)
#윈도우 방식 -> 첫 몇초는 판단 없음

def fill_blank(check_ts, labels, total_ts):
    def ts_generator():
        for t in total_ts:
            yield dateutil.parser.parse(t)

    def label_generator():
        for t, label in zip(check_ts, labels):
            yield dateutil.parser.parse(t), label

    g_ts = ts_generator()
    g_label = label_generator()
    final_labels = []

    try:
        current = next(g_ts)
        ts_label, label = next(g_label)
        while True:
            if current > ts_label:
                ts_label, label = next(g_label)
                continue
            elif current < ts_label:
                final_labels.append(0)
                current = next(g_ts)
                continue
            final_labels.append(label)
            current = next(g_ts)
            ts_label, label = next(g_label)
    except StopIteration:
        return np.array(final_labels, dtype=np.int8)

FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VAL_DF_RAW[TIMESTAMP_FIELD]))

#TaPR Score
tapr_score(FINAL_LABELS,ATTACK_LABELS)
