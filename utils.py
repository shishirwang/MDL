import numpy as np
from mnist import load
import pickle
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import combinations

def relu(x):
    y=x.copy()
    y[y<0]=0
    return y

def d_relu(x):
    y=x.copy()
    y[y<=0]=0
    y[y>0]=1
    return y

def softmax(x):
    x=x-np.max(x,axis=0)
    y=np.exp(x) 
    z=np.sum(y,axis=0)
    return y/z

class adam:
    beta1=0.9
    beta2=0.999
    epsilon=1e-8
    
    def __init__(self,theta):
        self.m=[0 for i in range(len(theta))]
        self.v=[0 for i in range(len(theta))]
        self.mhat=[0 for i in range(len(theta))]
        self.vhat=[0 for i in range(len(theta))]
        self.t=0
        self.theta=theta
        
    def update(self,gradient,learning_rate=0.001,decay=0):
        self.t+=1
        for i in range(len(gradient)):
            self.m[i]=adam.beta1*self.m[i]+(1-adam.beta1)*gradient[i]
            self.v[i]=adam.beta2*self.v[i]+(1-adam.beta2)*gradient[i]**2
            self.mhat[i]=self.m[i]/(1-adam.beta1**self.t)
            self.vhat[i]=self.v[i]/(1-adam.beta2**self.t)
            self.theta[i]=self.theta[i]-learning_rate*(self.mhat[i]/(self.vhat[i]**(1/2)+adam.epsilon)\
                                                       +decay*self.theta[i])
        return self.theta
