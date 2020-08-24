import os

import pandas as pd
import numpy as np


def load_processed(type,only_data=False):
    data=np.load('./processed/'+type+'.npy')
    if only_data:
        return data
    else:
        label=np.load('./processed/'+type+'_atk.npy')
        return data,label
