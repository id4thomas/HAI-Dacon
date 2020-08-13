import matplotlib.pyplot as plt
import numpy as np

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