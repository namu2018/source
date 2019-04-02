# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:11:46 2019

@author: ktm
"""

import nu
#%%mpy as np
import pandas as pd
from sklearn.datasets import load_boston
pd.set_option('display.float_format','{:.2f}'.format)
#%%
boston=load_boston()
print(type(boston))
print((boston.keys()))
#%%
print(boston.feature_names)
print(boston)
print(boston.target)
print(boston.data)
#%%
x=boston['data']
y=boston['target']
data=pd.DataFrame(x, columns=boston['feature_names'])
data['PRICE'] = y
#%%03. Gradient Descent
num_epoch=10000
learning_rate=0.00005
data.info()
x1=data['CRIM'].values
x2=data['ZN'].values
#%%
w1=np.random.uniform(low=0.0, high=1.0)
w2=np.random.uniform(low=0.0, high=1.0)
b=np.random.uniform(low=0.0, high=1.0)
for epoch in range(num_epoch):
    y_predict = x1*w1 + \
                x2*w2 +b
    err=np.abs(y_predict -y).mean()
    if err <3:
        break
    w1=w1-learning_rate*((y_predict-y)*x1).mean()
    w2=w2-learning_rate*((y_predict-y)*x2).mean()
    b=b-learning_rate* ((y_predict-y).mean())
    
    if epoch % 1000 == 0:
        print("epoch: {0:.2f} err:{1:.2f} w1:{2:.2f} w2:{3:.2f} b:{4:.2f}".format(epoch,err,w1,w2,b))
        #print("epoch", epoch,"err:",err,"w1:",w1,"w2:",w2,"b:",b)
print(w1,w2,err)
#%%

x1.shape