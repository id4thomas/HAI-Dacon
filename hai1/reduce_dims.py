import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA, KernelPCA,SparsePCA
import os

# from cuml.manifold import TSNE

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def plot2d(data,label,idx=[0,1],atk_front=False):
    fig=plt.figure()
    plt2d=fig.add_subplot(1,1,1)
    if atk_front:
        #safe
        s = plt2d.scatter(data[label==0,idx[0]], data[label==0,idx[1]], marker='x', color='y')
        #atk
        a = plt2d.scatter(data[label==1,idx[0]], data[label==1,idx[1]], marker='o', color='b')
    else:
        #atk
        a = plt2d.scatter(data[label==1,idx[0]], data[label==1,idx[1]], marker='o', color='b')
        #safe
        s = plt2d.scatter(data[label==0,idx[0]], data[label==0,idx[1]], marker='x', color='y')

    return fig

def plot3d(data,label,idx=[0,1,2],atk_front=False):
    fig=plt.figure()
    plt3d=fig.add_subplot(1,1,1,projection='3d')

    if atk_front:
        #safe
        plt3d.scatter(data[label==0,idx[0]], data[label==0,idx[1]],data[label==0,idx[2]], marker='x', color='y')
        #atk
        plt3d.scatter(data[label==1,idx[0]], data[label==1,idx[1]],data[label==1,idx[2]], marker='o', color='b')
    else:
        #safe
        plt3d.scatter(data[label==0,idx[0]], data[label==0,idx[1]],data[label==0,idx[2]], marker='x', color='y')
        #atk
        plt3d.scatter(data[label==1,idx[0]], data[label==1,idx[1]],data[label==1,idx[2]], marker='o', color='b')

    return fig

#1: atk, 0: safe
def run_reduc(data,label,d_type,l_type,reduc_type='pca',reduc=None):
    # reduc_type='tsne'
    # reduc = TSNE(n_jobs=32)
    
    reduc_type='spca'
    if reduc is None:
        reduc=PCA(n_components=20)
        # reduc=KernelPCA(n_components=3,kernel='rbf')
        # reduc=SparsePCA(n_components=10)
        # tsne=TSNE(n_components=2)
        data=reduc.fit_transform(data)
    else:
        data=reduc.transform(data)

    print('Reduc Complete')
    vr=reduc.explained_variance_ratio_
    print(reduc.explained_variance_ratio_)
    print(np.cumsum(np.array(vr)))
    df_converted=pd.DataFrame(data)
    df_converted=pd.concat([df_converted,pd.DataFrame(label)],axis=1)
    # df_converted.to_csv('./hai1/reduc/{}_{}.csv'.format(reduc_type,d_type),header=None,index=False)

    print('Reduc Save Complete')
    # Load from Saved

    # data=pd.read_csv('./hai1/tsne/tsne_'+d_type+'.csv',header=None)
    # data=np.array(data.iloc[:,0:2])
    
    # fig=plot2d(data,label,idx=[0,1],atk_front=True)
    # fig.savefig('./plot/{}_{}_idx1_2.png'.format(d_type,reduc_type))

    # fig=plot2d(data,label,idx=[0,2],atk_front=True)
    # fig.savefig('./plot/{}_{}_idx1_3.png'.format(d_type,reduc_type))

    # fig=plot2d(data,label,idx=[1,2],atk_front=True)
    # fig.savefig('./plot/{}_{}_idx2_3.png'.format(d_type,reduc_type))

    # fig=plot3d(data,label,idx=[0,1,2],atk_front=True)
    # fig.savefig('./plot/{}_{}_3d.png'.format(d_type,reduc_type))
    

    # plt.savefig('./plot/{}_{}.png'.format(d_type,reduc_type))
    return reduc

def load_data(d_type,l_type):
    #print(df.head)

    #data
    # data=np.load('./hai1/processed/{}.npy'.format(d_type))
    # label=np.load('./hai1/processed/{}_{}.npy'.format(d_type,l_type))

    #Latent
    # data=np.load('./model_hid/{}_hids.npy'.format(d_type))#hids,dist
    # label=np.load('./model_hid/{}_{}.npy'.format(d_type,l_type))

    data=np.load('./data/processed/{}.npy'.format(d_type))#hids,dist
    label=np.load('./data/processed/{}_{}.npy'.format(d_type,l_type))

    #print(df.head)
    return data,label

l_type='atk'
d_type='train'

train_data,train_label=load_data(d_type,'atk')
print("Running PCA Train")
print('atk',len(train_data[train_label==1]))
print('safe',len(train_data[train_label==0]))
reduc=run_reduc(train_data,train_label,d_type,l_type)

d_type='test'
test_data,test_label=load_data(d_type,'atk')
print("Running PCA Test")
print('atk',len(test_data[test_label==1]))
print('safe',len(test_data[test_label==0]))
run_reduc(test_data,test_label,d_type,l_type,reduc=reduc)
