import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from sklearn import metrics

#TaPR
from TaPR_pkg import etapr

def tapr_score(predictions,label):
    TaPR = etapr.evaluate(anomalies=label, predictions=predictions)
    print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
    print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

def dist_graph(anomaly_score,atk,piece=2):
    #Plot point of attack & recon loss at the point
    l = anomaly_score.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, anomaly_score[L:R])
        if len(anomaly_score[L:R]) > 0:
            peak = max(anomaly_score[L:R])
            axs[i].plot(xticks, atk[L:R] * peak * 0.3)
    # plt.savefig('./baseline/baseline_dist.png')
    return fig

def make_roc(loss,label,ans_label=1):
    normalDataLoss=[]
    attackDataLoss=[]

    for i in range(len(loss)):
        if(label[i]==ans_label):
            attackDataLoss.append(loss[i])
        else:
            normalDataLoss.append(loss[i])

    print("Normal data loss(%d): %f" % (len(normalDataLoss), np.average(np.array(normalDataLoss))))
    print("Attack data loss(%d): %f" % (len(attackDataLoss), np.average(np.array(attackDataLoss))))

    allDataLoss = normalDataLoss+attackDataLoss
    print("Sum : ", len(allDataLoss))
    print(len(normalDataLoss))
    allLabel = [0]*len(normalDataLoss)+[1]*len(attackDataLoss)

    fpr, tpr, thresholds = metrics.roc_curve(np.array(allLabel), np.array(allDataLoss), pos_label=1, drop_intermediate=False)

    fig=plt.figure()
    roc=fig.add_subplot(1,1,1)
    lw = 2
    roc.plot(fpr, tpr, color='darkorange', lw=lw, )
    roc.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    roc.set_xlim([0.0, 1.0])
    roc.set_ylim([0.0, 1.05])
    roc.set_xlabel('False Positive Rate')
    roc.set_ylabel('True Positive Rate')
    roc.set_title('ROC-curve')
    roc.legend(loc="lower right")
    # fig.savefig('./plot/roc_curve.png', dpi=80)

    print('AUC Score',metrics.roc_auc_score(np.array(allLabel), np.array(allDataLoss)))
    return fig

def make_pr(pred,label):
    precision, recall, thresholds = metrics.precision_recall_curve(label, pred)
    fig=Figure()
    fig.plot(recall, precision, marker='.', label='Logistic')

    fig.xlabel('Recall')
    fig.ylabel('Precision')
    fig.title('Precision-Recall Curve')
    fig.legend()
    return fig
