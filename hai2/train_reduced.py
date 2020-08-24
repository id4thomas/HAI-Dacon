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
import argparse

from data_utils import *
from reduce_utils import *

parser = argparse.ArgumentParser(description='HAI Train Params')
parser.add_argument('--ep', type=int, default=32,
                    help='Max Epochs')
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

MAX_EPOCHS = args.ep
BATCH_SIZE = args.batch

MODEL_NAME=args.m_name
REDUC_TYPE=args.r_type

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

#Load data from processed
#Load dataset - for time
TRAIN_DATASET = sorted([x for x in Path('./data/training/').glob("*.csv")])
# VAL_DATASET = sorted([x for x in Path('./data/validation/').glob("*.csv")])
TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
# VAL_DF_RAW = dataframe_from_csvs(VAL_DATASET)
TIMESTAMP_FIELD = "time"
ATTACK_FIELD = "attack"

NUM_COMPONENTS=25
x_train_r=load_processed('train_'+REDUC_TYPE,only_data=True)

#Reduce Data Dimensions

# x_train_r,reduc=train_reduc(x_train,'grp',n_c=NUM_COMPONENTS)
# # x_val_r=reduc.transform(x_val)
# print("Reduced Data to {}".format(x_train_r.shape))

#Make Torch Data

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


HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], x_train_r, stride=10)

#Model
if MODEL_NAME=='baseline':
    from baseline_model import HAIModel
    loss_fn = torch.nn.MSELoss()

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
print("Built Model")

def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())

    epochs = range(n_epochs)
    best = {"loss": sys.float_info.max}
    loss_history = []
    for e in epochs:
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            if MODEL_NAME=='baseline':
                guess = MODEL(given)
                answer = batch["answer"].cuda()
                loss = loss_fn(answer, guess)
            elif MODEL_NAME=='vae':
                guess,mu,var = MODEL(given)
                answer = batch["answer"].cuda()
                loss = MODEL.loss_fn(mu,var,answer, guess)

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


######## Train #######
MODEL.train()

BEST_MODEL, LOSS_HISTORY = train(HAI_DATASET_TRAIN, MODEL, BATCH_SIZE, MAX_EPOCHS)

print(BEST_MODEL["loss"], BEST_MODEL["epoch"])

with open("./model/{}.pt".format(MODEL_NAME), "wb") as f:
    torch.save(
        {
            "state": BEST_MODEL["state"],
            "best_epoch": BEST_MODEL["epoch"],
            "loss_history": LOSS_HISTORY,
        },
        f,
    )

######## Train #######
