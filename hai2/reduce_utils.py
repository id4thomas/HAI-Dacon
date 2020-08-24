import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, KernelPCA,SparsePCA,FastICA,TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from perf_utils import *
from data_utils import *

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report

ATK=1
SAFE=0

#Plot variance ratio for pca
def plot_var(data,reduc_type,kernel='rbf',n_c=8):
    _,reduc=train_reduc(data,reduc_type=reduc_type,kernel=kernel,n_c=n_c)
    vr=np.array(reduc.explained_variance_ratio_)
    print(reduc.explained_variance_ratio_)
    print(np.cumsum(vr))

    vrfig=plt.figure()
    pltauc=vrfig.add_subplot(1,1,1)
    pltauc.plot(range(vr.shape[0]),vr)
    pltauc.set_title('Variance Ratio')
    # fig.savefig('./plot/{}_vr.png'.format(reduc_type))
    # plt.clf()

    cvrfig=plt.figure()
    pltauc=cvrfig.add_subplot(1,1,1)
    pltauc.plot(range(vr.shape[0]),np.cumsum(vr))
    pltauc.set_title('Accumulated Variance Ratio')
    # fig.savefig('./plot/{}_cum_vr.png'.format(reduc_type))
    return vrfig,cvrfig

def train_reduc(data,reduc_type='pca',kernel='rbf',n_c=8,eps=0.01,random_state=2020):
    if reduc_type=='pca':
        reduc=PCA(n_components=n_c)
    elif reduc_type=='spca':
        reduc=SparsePCA(n_components=n_c)
    elif reduc_type=='kpca':
        reduc=KernelPCA(n_components=n_c,kernel=kernel)
    elif reduc_type=='ica':
        reduc=FastICA(n_components=n_c)
    elif reduc_type=='grp':
        reduc=GaussianRandomProjection(n_components=n_c,eps=eps,random_state=random_state)
    elif reduc_type=='srp':
        reduc=SparseRandomProjection(n_components=n_c,density='auto',eps=eps,dense_output=True,random_state=random_state)

    reduced=reduc.fit_transform(data)
    print('Reduc Complete')
    return reduced,reduc

def test_reduc(data,label,reduc,reduc_type,dis='l1'):
    #Apply Reduc
    data_reduc=reduc.transform(data)
    #Recon
    if reduc_type in ['pca','kpca','ica']:
        #If inverse available
        data_recon=reduc.inverse_transform(data_reduc)
    elif reduc_type=='spca':
        #spca
        data_recon=np.array(data_reduc).dot(reduc.components_)+np.array(data.mean(axis=0))
    elif reduc_type=='grp':
        data_recon=np.array(data_reduc).dot(reduc.components_)
    elif reduc_type=='srp':
        data_recon=np.array(data_reduc).dot(reduc.components_.todense())
    else:
        pass

    #Calculate Recon Loss
    if dis=='l1':
        dist=np.mean(np.abs(data-data_recon),axis=1)
    elif dis=='l2':
        dist=np.mean(np.square(data - data_recon),axis=1)

    roc,auc,desc=make_roc(dist,label,ans_label=ATK,make_desc=True)
    return roc,auc,desc

if __name__ == "__main__":
    #instance based
    x_train=load_processed('train',only_data=True)
    x_val,y_val=load_processed('val')
    aucs=[]
    #pca l1: 27 0.98560 , l2: 25 0.98661
    #ica l1: , l2:
    #grp l1: 25 0.77256 , l2:6 0.76441
    #srp l1: 24 0.88013  l2: 24 0.88952
    n_c=25
    _,reduc=train_reduc(x_train,reduc_type='pca',n_c=n_c)
    roc,auc,desc=test_reduc(x_val,y_val,reduc,'pca',dis='l2')
    roc.savefig('./plot/pca_25_l2.png')
    desc.to_csv('./desc/pca_25_l2.csv')
    # for n_c in range(5,33):
    #     _,reduc=train_reduc(x_train,reduc_type='ica',n_c=n_c)
    #     roc,auc,desc=test_reduc(x_val,y_val,reduc,'ica',dis='l2')
    #     aucs.append(auc)
    # print("Best {:.5f} at {}".format(max(aucs),5+aucs.index(max(aucs))))
    
