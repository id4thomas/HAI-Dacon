import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#My Utils
from utils.data_utils import *

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

#Torch Dataset
HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=STRIDE)


#Train Function
def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lr=1e-3
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    epochs = range(n_epochs)
    best = {"loss": sys.float_info.max}
    loss_history = []
    loss_fn = torch.nn.MSELoss()

    for e in epochs:
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            guess = model(given)

            loss = loss_fn(answer, guess)
            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()

        print('Epoch {} Loss {:.6f}'.format(e+1,epoch_loss))
        loss_history.append(epoch_loss)

        #For validation Loss
        #with torch.no_grad():
        # epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1
    return best, loss_history

######## TRAIN #########
MAX_EPOCHS_GRU=args.ep
BATCH_SIZE=args.batch

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

BEST_MODEL, LOSS_HISTORY = train(HAI_DATASET_TRAIN, MODEL, BATCH_SIZE, MAX_EPOCHS_GRU)
print(BEST_MODEL["loss"], BEST_MODEL["epoch"])

with open("./model/gru{}.pt".format(gru_name), "wb") as f:
    torch.save(
        {
            "state": BEST_MODEL["state"],
            "best_epoch": BEST_MODEL["epoch"],
            "loss_history": LOSS_HISTORY,
        },
        f,
    )
