import pandas as pd
import numpy as np
import math
#import matplotlib.pyplot 

from sklearn.model_selection import ShuffleSplit

from collections import Counter
from imblearn.over_sampling import SMOTE
from numpy import where

from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from sklearn.metrics import r2_score,mean_absolute_error,precision_score,recall_score,roc_auc_score,roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import MinMaxScaler  
from sklearn.manifold import TSNE 
#import matplotlib.pyplot as plt 
#import seaborn as sns 
from keras.layers import Input, Dense 
from keras.models import Model, Sequential 
from keras import regularizers 

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
from copy import copy
import time
#from torch import *
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
#from torch.model import VAE
#from gtex_loader import get_gtex_dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('Dataset_Dual Phase Steel.csv')
#print(df)
df=df.to_numpy()
#print(df)
#print(df.shape)
df=df[:,1:23]
#print(df)
#print(df.shape)

x=df[:,0:20]            #Input Parameters(Composition, Thickness and Intercritical Annealing Schedule)
#print(x)
#print(x.shape)

y=df[:,20:23]           #Output Parameters(Martensite and ferrite volume fraction)
#print(y)
#print(y.shape)

#Indices for shuffle split
index_train,index_test=[],[]
ss=ShuffleSplit(n_splits=1,test_size=0.3,random_state=0)
for train_index,test_index in ss.split(x):
    #print("%s%s"%(train_index,test_index))
    index_train.append(train_index)
    index_test.append(test_index)
#print(index_train)
#print(index_test)
index_train=np.array(index_train)
index_test=np.array(index_test)
#print(index_train.shape)
#print(index_test)

#print(index_train[0][1])

#Obtaining train_test split
x_train,x_test,y_train,y_test=[],[],[],[]

for i in range(0,index_train.shape[1],1):
    x_train.append(x[index_train[0][i]:index_train[0][i],:])
    y_train.append(y[index_train[0][i]])
        
for i in range(index_test.shape[0]):
    for j in range(index_test.shape[1]):
        x_test.append(x[index_test[i][j]])
        y_test.append(y[index_test[i][j]])

x_train=np.array(x_train)                      #Input train 
x_test=np.array(x_test)                        #Input test 
#print(x_train)
#print(x_test)
#print(x_train.shape)
#print(x_test.shape)

y_train=np.array(y_train)                      #Output Train 
y_test=np.array(y_test)                        #Output Test  
#print(y_train)
#print(y_test)
#print(y_train.shape)
#print(y_test.shape)

writer=SummaryWriter('runs/lgbm')

def idx2onehot(idx,n):
    assert torch.max(idx).item()<n
    if idx.dim()==1:
        idx=idx.unsqueeze(1)
    onehot=torch.zeros(idx.size(0),n)
    onehot.scatter_(1,idx,1)
    return onehot


class VAE(nn.Module):

    def __init__(self,encoder_layer_sizes,latent_size,decoder_layer_sizes,conditional=False,num_labels=0):
        super().__init__()
        if conditional:
            assert num_labels>0
        assert type(encoder_layer_sizes)==list
        assert type(latent_size)==int
        assert type(decoder_layer_sizes)==list
        self.latent_size=latent_size
        self.encoder=Encoder(encoder_layer_sizes,latent_size,conditional,num_labels)
        self.decoder=Decoder(decoder_layer_sizes,latent_size,conditional,num_labels)

    def forward(self,x,c=None):
        view_size=1000
        if x.dim()>2:
            x=x.view(-1,view_size)
        batch_size=x.size(0)
        means,log_var=self.encoder(x,c)
        std=torch.exp(0.5*log_var)
        eps=torch.randn([batch_size,self.latent_size])
        z=eps*std+means
        recon_x=self.decoder(z,c)
        return recon_x,means,log_var,z

    def inference(self,n=10,c=None):
        batch_size=n
        z=torch.randn([batch_size,self.latent_size])
        recon_x=self.decoder(z,c)
        return recon_x

    def embedding(self,x,c=None):
        view_size=1000
        if x.dim()>2:
            x=x.view(-1,view_size)
        means,log_var=self.encoder(x,c)
        return means,log_var


class Encoder(nn.Module):
    
    def __init__(self,layer_sizes,latent_size,conditional,num_labels):
        super().__init__()
        self.conditional=conditional
        if self.conditional:
            layer_sizes[0]+=num_labels
        self.MLP=nn.Sequential()
        for i,(in_size,out_size) in enumerate(zip(layer_sizes[:-1],layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i),module=nn.Linear(in_size,out_size))
            self.MLP.add_module(name="A{:d}".format(i),module=nn.ReLU())
        self.linear_means=nn.Linear(layer_sizes[-1],latent_size)
        self.linear_log_var=nn.Linear(layer_sizes[-1],latent_size)
        
    def forward(self,x,c=None):
        if self.conditional:
            c=idx2onehot(c,n=6)
            x=torch.cat((x,c),dim=-1)
        x=self.MLP(x)
        means=self.linear_means(x)
        log_vars=self.linear_log_var(x)
        return means,log_vars

    
class Decoder(nn.Module):

    def __init__(self,layer_sizes,latent_size,conditional,num_labels):
        super().__init__()
        self.MLP=nn.Sequential()
        self.conditional=conditional
        if self.conditional:
            input_size=latent_size+num_labels
        else:
            input_size=latent_size
        for i,(in_size,out_size) in enumerate(zip([input_size]+layer_sizes[:-1],layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i),module=nn.Linear(in_size,out_size))
            if i+1<len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i),module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid",module=nn.Sigmoid())

    def forward(self,z,c):
        if self.conditional:
            c=idx2onehot(c,n=6)
            z=torch.cat((z,c),dim=-1)
        x=self.MLP(z)
        return x
                                     
writer=SummaryWriter('runs/lgbm')


def main(args):
    torch.manual_seed(args.seed)
    latest_loss = torch.tensor(1)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = time.time()

    (X_train, Y_train), (X_test, Y_test), scaled_df_values, gene_names, Y = get_gtex_dataset()

    le = LabelEncoder()
    le.fit(Y_train)
    train_targets = le.transform(Y_train)
    test_targets = le.transform(Y_test)
    print(le.classes_)

    train_target = torch.as_tensor(train_targets)
    train = torch.tensor(X_train.astype(np.float32))
    train_tensor = TensorDataset(train, train_target)
    data_loader = DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)

    test_target = torch.as_tensor(test_targets)
    test = torch.tensor(X_test.astype(np.float32))
    test_tensor = TensorDataset(test, test_target)
    test_loader = DataLoader(dataset=test_tensor, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        view_size = 1000
        ENTROPY = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, view_size), x.view(-1, view_size), reduction='sum')
        HALF_LOG_TWO_PI = 0.91893
        MSE = torch.sum((x - recon_x)**2)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        gamma_square = 0
        if torch.eq(latest_loss, torch.tensor(1)):
            gamma_square = MSE
        else:
            gamma_square = min(MSE, latest_loss.clone())
        #print(gamma_square)
        #print(MSE,KLD)
       # return {'GL': (MSE/(2*gamma_square.clone()) + torch.log(torch.sqrt(gamma_square)) + HALF_LOG_TWO_PI + KLD) / x.size(0), 'MSE': MSE}
        #return {'GL': (50*MSE + KLD) / x.size(0), 'MSE': MSE}
        beta = 0.9
        return {'GL': (ENTROPY + KLD) / x.size(0), 'MSE': MSE}
        #return {'GL': (ENTROPY + 50*KLD) / x.size(0), 'MSE': MSE}

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=6 if args.conditional else 0).to(device)

    dataiter = iter(data_loader)
    genes, labels = dataiter.next()
    writer.add_graph(vae, genes)
    writer.close()

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):
        train_loss = 0
        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            multiple_losses = loss_fn(recon_x, x, mean, log_var)
            loss = multiple_losses['GL'].clone()
            train_loss += loss

            latest_loss = multiple_losses['MSE'].detach()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            logs['loss'].append(loss.item())



            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.arange(0, 6).long().unsqueeze(1)
                    x = vae.inference(n=c.size(0), c=c)
                else:
                    x = vae.inference()


    with torch.no_grad():
        for epoch in range(args.epochs):
            test_loss = 0
            for iteration, (x, y) in enumerate(test_loader):
                recon_x, mean, log_var, z = vae(x, y)
                test_loss = loss_fn(recon_x, x, mean, log_var)['GL']

                if iteration == len(test_loader) - 1:
                    print('====> Test set loss: {:.4f}'.format(test_loss.item()))

    with torch.no_grad():
        y_synthetic = []
        x_synthetic = []
        for i in range(6):
            c = np.array([i for j in range(2000)])
            x = vae.inference(n=len(c), c=c)
            x_synthetic += list(x.detach().numpy()[:,:1000])
            y_synthetic += list(np.ravel(le.inverse_transform(c)))

        x_df = pd.DataFrame(x_synthetic, columns=gene_names).to_csv('data/expressions_synthetic_2000.csv', index=False)
        y_df = pd.DataFrame(y_synthetic, columns=['Age']).to_csv('data/samples_synthetic_2000.csv', index=False)

    check_reconstruction_and_sampling_fidelity(vae, scaled_df_values, Y, gene_names)

def check_reconstruction_and_sampling_fidelity(vae_model,scaled_df_values, Y, gene_names):
    # get means of original columns based on 100 first rows
    genes_to_validate = 40
    original_means = np.mean(scaled_df_values, axis=0)
    original_vars = np.var(scaled_df_values, axis=0)

    #mean, logvar = vae_model.encode(scaled_df_values, Y)
    #z = vae_model.reparameterize(mean, logvar)

    #plot_dataset_in_3d_space(z, y_values)

    #x_decoded = vae_model.decode(z, Y)

    #decoded_means = np.mean(x_decoded, axis=0)
    #decoded_vars = np.var(x_decoded, axis=0)

    with torch.no_grad():
        number_of_samples = 2000
        class_0 = [0 for i in range(number_of_samples)]
        class_1 = [1 for i in range(number_of_samples)]
        class_2 = [2 for i in range(number_of_samples)]
        class_3 = [3 for i in range(number_of_samples)]
        class_4 = [4 for i in range(number_of_samples)]
        class_5 = [5 for i in range(number_of_samples)]
        all_samples = np.array(class_0 + class_1 + class_2 + class_3 + class_4 + class_5)
        c = torch.from_numpy(all_samples)
        print(c)
        x = vae_model.inference(n=len(all_samples), c=c)
        print(x)

    sampled_means = np.mean(x.detach().numpy(), axis=0)
    sampled_vars = np.var(x.detach().numpy(), axis=0)

    #abs_dif = np.divide(np.sum(np.absolute(scaled_df_values - x_decoded), axis=0), df_values.shape[0])
    #abs_dif_by_mean = np.divide(np.divide(np.sum(np.absolute(df_values - x_decoded), axis=0), df_values.shape[0]), original_means)

   # mean_deviations = np.absolute(original_means - decoded_means)
    #print(pd.DataFrame(list(zip(df_columns, mean_deviations)), columns=['Gene', 'Mean Dif']).sort_values(by=['Mean Dif'], ascending=False))

    #print(predictions[0][:10])
    #print(df_values[5][:10])
    #print(x_decoded[5][:10])

    plot_reconstruction_fidelity(original_means[:genes_to_validate], sampled_means[:genes_to_validate], metric_name='Mean', df_columns=gene_names)
    plot_reconstruction_fidelity(original_vars[:genes_to_validate], sampled_vars[:genes_to_validate], metric_name='Variance', df_columns=gene_names)
    #plot_reconstruction_fidelity(abs_dif[:genes_to_validate], metric_name='Absolute Difference (Sum by samples)')
    #plot_reconstruction_fidelity(abs_dif_by_mean[:genes_to_validate], metric_name='Absolute Difference (Sum by samples, Divided by gene Mean)')

    #print('Sum of Mean difference by gene: ', np.mean(np.absolute(original_means - decoded_means)))
    #print('Sum of Absolute difference by gene: ', np.mean(np.sum(np.absolute(df_values - x_decoded), axis=0) / df_values.shape[0]))
    #print('Sum of Absolute difference divided by gene Mean: ', np.mean(abs_dif_by_mean))

def plot_reconstruction_fidelity(original_values, sampled_values=[], metric_name='', df_columns=[]):
    n_groups = len(original_values)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    if len(sampled_values) > 0:
        plt.bar(index, original_values, bar_width, alpha=opacity, color='b', label='Original')
        plt.bar(index + bar_width, sampled_values, bar_width, alpha=opacity, color='g', label='Reconstructed')
        plt.title('Original VS Reconstructed ' + metric_name)
        plt.xticks(index + bar_width, list(df_columns)[:n_groups], rotation=90)
        plt.ylabel(metric_name + ' Expression Level (Scaled)')
        plt.legend()
    else:
        plt.bar(index, original_values, bar_width, alpha=opacity, color='b')
        plt.title(metric_name)
        plt.xticks(index, list(df_columns)[:n_groups], rotation=90)
        plt.ylabel('Expression Level (Scaled)')
        plt.legend()

    plt.xlabel('Gene Name')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1000, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 512, 1000])
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
  
for i in range(x_test.shape[0]):
    tempx,tempy=[],[]
    for j in range(x_train.shape[0]):
        if mixture_test[i]==mixture_train[j]:
            tempx.append(x_train[j])
            tempy.append(y_train[j])
    tempx=np.array(tempx)
    tempy=np.array(tempy)
    #print(tempx)
    #print(tempx.shape)
    #print(tempy)
    #print(tempy.shape)
  
y_train=y_train.astype('int32')
y_test=y_test.astype('int32')

# transform the dataset
oversample=SMOTE()
x1_train,y1_train=oversample.fit_resample(x_train,y_train)
#print(x_train.shape)
#print(y_train.shape)

clf=RandomForestClassifier(max_depth=5,random_state=0)
clf.fit(x1_train,y1_train)
y_clf=clf.predict(x_test)
'''for i in range(y_test.shape[0]):
    print('{}   {}'.format(y_test[i],y_clf[i]))'''
    
print("Precision={}".format(precision_score(y_test,y_clf,average='weighted')))
print("Recall={}".format(recall_score(y_test,y_clf,average='weighted')))
print("Area under ROC={}".format(roc_auc_score(y_test,y_clf)))
fpr,tpr,thresholds=roc_curve(y_test,y_clf,pos_label=1)
eer=brentq(lambda x:1.0-x-interp1d(fpr,tpr)(x),0.0,1.0)
print("EER={}".format(eer))

ns_probs=[0 for _ in range(len(y_test))]
lr_probs=clf.predict_proba(x_test)
lr_probs=lr_probs[:,1]
ns_auc=roc_auc_score(y_test,ns_probs)
lr_auc=roc_auc_score(y_test,lr_probs)
print('No Skill: ROC AUC=%.3f'%(ns_auc))
print('Logistic: ROC AUC=%.3f'%(lr_auc))
ns_fpr,ns_tpr,_=roc_curve(y_test,ns_probs)
lr_fpr,lr_tpr,_=roc_curve(y_test,lr_probs)
plt.plot(ns_fpr,ns_tpr,linestyle='--',label='No Skill')
plt.plot(lr_fpr,lr_tpr,marker='.',label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
  
#Prediction of phase fractions

x1_train=np.column_stack((x_train,F_train))
x1_test=np.column_stack((x_test,F_test))

#NuSVR
sv=make_pipeline(StandardScaler(),NuSVR(C=0.9999,nu=1,max_iter=100,gamma=0.0888))
#RFR
rf=RandomForestRegressor(n_estimators=50,max_depth=3,random_state=0,min_samples_split=2)
#GBR
gb=GradientBoostingRegressor(random_state=0,learning_rate=0.09,n_estimators=150,min_samples_split=18,min_samples_leaf=4,alpha=0.5,max_leaf_nodes=10)

sv.fit(x1_train,F_train)
rf.fit(x1_train,F_train)
gb.fit(x1_train,F_train)

yA1=sv.predict(x1_test)
yA2=rf.predict(x1_test)
yA3=gb.predict(x1_test)

rA1=r2_score(F_test,yA1)
rA2=r2_score(F_test,yA2)
rA3=r2_score(F_test,yA3)

print("Regressor                  Accuracy")
print("NuSVC                      {}".format(rA1))
print("RF                         {}".format(rA2))
print("GB                         {}".format(rA3))

'''y_M,y_P=[],[]
for i in range(yA4.shape[0]):
    if y_clf[i]==0:
        y_M.append(1-yA4[i])
        y_P.append(0)
    else:
        y_M.append(0)
        y_P.append(1-yA4[i])'''
        
'''y_M=np.array(y_M)
y_P=np.array(y_P)
print("Accuracy in martensite phase fraction prediction={}".format(r2_score(M_test,y_M)))
print("Accuracy in pearlite phase fraction prediction={}".format(r2_score(P_test,y_P)))'''

fig,ax=plt.subplots()
ax.scatter(F_test,yA1)
ax.scatter(F_test,yA2)
ax.scatter(F_test,yA3)
ax.plot([F_test.min(),F_test.max()],[yA3.min(),yA3.max()],'k--',lw=1)
ax.set_xlabel('Measured Ferrite Phase Fraction')
ax.set_ylabel('Predicted Ferrite Phase Fraction')
plt.show()

fig,ax=plt.subplots()
ax.scatter(F_test,yA3)
ax.plot([F_test.min(),F_test.max()],[yA3.min(),yA3.max()],'k--',lw=1)
ax.set_xlabel('Measured Ferrite Phase Fraction')
ax.set_ylabel('Predicted Ferrite Phase Fraction')
plt.show()

fig,ax=plt.subplots()
ax.scatter(M_test,y_M)
ax.plot([M_test.min(),M_test.max()],[y_M.min(),y_M.max()],'k--',lw=1)
ax.set_xlabel('Measured Martensite Phase Fraction')
ax.set_ylabel('Predicted Martensite Phase Fraction')
plt.show()

'''fig,ax=plt.subplots()
ax.scatter(P_test,y_P)
ax.plot([P_test.min(),P_test.max()],[y_P.min(),y_P.max()],'k--',lw=1)
ax.set_xlabel('Measured Pearlite Phase Fraction')
ax.set_ylabel('Predicted Pearlite Phase Fraction')
plt.show()'''
  
#Prediction of Total Elongation

x2_train=np.concatenate((C_train, PF_train), axis=1)
x2_test=np.concatenate((C_test, PF_test), axis=1)

#print(x2_train.shape)
#print(x2_test.shape)

x2_train=np.column_stack((x2_train,TE_train))
x2_test=np.column_stack((x2_test,TE_test))

#NuSVR
sv=make_pipeline(StandardScaler(),NuSVR(C=0.9999,nu=1,max_iter=100,gamma=0.0888))
#RFR
rf=RandomForestRegressor(n_estimators=50,max_depth=3,random_state=0,min_samples_split=2)
#GBR
gb=GradientBoostingRegressor(random_state=0,learning_rate=0.09,n_estimators=150,min_samples_split=18,min_samples_leaf=4,alpha=0.5,max_leaf_nodes=10)

sv.fit(x2_train,TE_train)
rf.fit(x2_train,TE_train)
gb.fit(x2_train,TE_train)

yA1=sv.predict(x2_test)
yA2=rf.predict(x2_test)
yA3=gb.predict(x2_test)

rA1=r2_score(TE_test,yA1)
rA2=r2_score(TE_test,yA2)
rA3=r2_score(TE_test,yA3)

print("Regressor                  Accuracy")
print("NuSVC                      {}".format(rA1))
print("RF                         {}".format(rA2))
print("GB                         {}".format(rA3))

fig,ax=plt.subplots()
ax.scatter(TE_test,yA1)
ax.scatter(TE_test,yA2)
ax.scatter(TE_test,yA3)
ax.plot([TE_test.min(),TE_test.max()],[yA2.min(),yA2.max()],'k--',lw=1)
ax.set_xlabel('Measured Total Elongation')
ax.set_ylabel('Predicted Total Elongation')
plt.show()

fig,ax=plt.subplots()
ax.scatter(TE_test,yA2)
ax.plot([TE_test.min(),TE_test.max()],[yA2.min(),yA2.max()],'k--',lw=1)
ax.set_xlabel('Measured Total Elongation')
ax.set_ylabel('Predicted Total Elongation')
plt.show()
  
#Prediction of Yield Strength

x2_train=np.column_stack((x2_train,YS_train))
x2_test=np.column_stack((x2_test,YS_test))

#NuSVR
sv=make_pipeline(StandardScaler(),NuSVR(C=0.9999,nu=1,max_iter=100,gamma=0.0888))
#RFR
rf=RandomForestRegressor(n_estimators=50,max_depth=3,random_state=0,min_samples_split=2)
#GBR
gb=GradientBoostingRegressor(random_state=0,learning_rate=0.09,n_estimators=150,min_samples_split=18,min_samples_leaf=4,alpha=0.5,max_leaf_nodes=10)

sv.fit(x2_train,YS_train)
rf.fit(x2_train,YS_train)
gb.fit(x2_train,YS_train)

yA1=sv.predict(x2_test)
yA2=rf.predict(x2_test)
yA3=gb.predict(x2_test)

rA1=r2_score(YS_test,yA1)
rA2=r2_score(YS_test,yA2)
rA3=r2_score(YS_test,yA3)

print("Regressor                  Accuracy")
print("NuSVC                      {}".format(rA1))
print("RF                         {}".format(rA2))
print("GB                         {}".format(rA3))

fig,ax=plt.subplots()
ax.scatter(YS_test,yA1)
ax.scatter(YS_test,yA2)
ax.scatter(YS_test,yA3)
ax.plot([YS_test.min(),YS_test.max()],[yA3.min(),yA3.max()],'k--',lw=1)
ax.set_xlabel('Measured Yield Strength')
ax.set_ylabel('Predicted Yield Strength')
plt.show()

fig,ax=plt.subplots()
ax.scatter(YS_test,yA3)
ax.plot([YS_test.min(),YS_test.max()],[yA3.min(),yA3.max()],'k--',lw=1)
ax.set_xlabel('Measured Yield Strength')
ax.set_ylabel('Predicted Yield Strength')
plt.show()
  
#Prediction of Ultimate Tensile Strength

x2_train=np.column_stack((x_train,UTS_train))
x2_test=np.column_stack((x_test,UTS_test))

#NuSVR
sv=make_pipeline(StandardScaler(),NuSVR(C=0.9999,nu=1,max_iter=100,gamma=0.0888))
#RFR
rf=RandomForestRegressor(n_estimators=50,max_depth=3,random_state=0,min_samples_split=2)
#GBR
gb=GradientBoostingRegressor(random_state=0,learning_rate=0.09,n_estimators=150,min_samples_split=18,min_samples_leaf=4,alpha=0.5,max_leaf_nodes=10)

sv.fit(x2_train,UTS_train)
rf.fit(x2_train,UTS_train)
gb.fit(x2_train,UTS_train)

yA1=sv.predict(x2_test)
yA2=rf.predict(x2_test)
yA3=gb.predict(x2_test)

rA1=r2_score(UTS_test,yA1)
rA2=r2_score(UTS_test,yA2)
rA3=r2_score(UTS_test,yA3)

print("Regressor                  Accuracy")
print("NuSVC                      {}".format(rA1))
print("RF                         {}".format(rA2))
print("GB                         {}".format(rA3))

fig,ax=plt.subplots()
ax.scatter(UTS_test,yA1)
ax.scatter(UTS_test,yA2)
ax.scatter(UTS_test,yA3)
ax.plot([UTS_test.min(),UTS_test.max()],[yA3.min(),yA3.max()],'k--',lw=1)
ax.set_xlabel('Measured Ultimate Yield Strength')
ax.set_ylabel('Predicted Ultimate Yield Strength')
plt.show()

fig,ax=plt.subplots()
ax.scatter(UTS_test,yA3)
ax.plot([UTS_test.min(),UTS_test.max()],[yA3.min(),yA3.max()],'k--',lw=1)
ax.set_xlabel('Measured Ultimate Yield Strength')
ax.set_ylabel('Predicted Ultimate Yield Strength')
plt.show()
  
#Prediction of composition given mechanical properties

pca=PCA(n_components=3,svd_solver='full')
new=pca.fit_transform(C_train)
#print(new)
#print(new.shape)
#print(x2_train.shape)
#print(mean_absolute_error(new,x2_train))
t=np.transpose(MP_train)
#print(t.shape)
a=np.dot(t,MP_train)
a=np.array(a,dtype='float')
l=[[1,0,0],[0,1,0],[0,0,1]]
l=np.array(l)
#print(l.shape)
aa=np.add(a,l)
aaa=np.linalg.inv(aa)
b=np.dot(aaa,t)
#print(a.shape)
bb=np.dot(b,new)
#print(b)
#print(bb.shape)
bbb=np.dot(MP_test,bb)
#print(bbb.shape)
#print(bbb)
new1=pca.inverse_transform(bbb)
#print(new1.shape)
#print(new1)
print("Mean Absolute Error={}".format(mean_absolute_error(new1,C_test)))

fig,ax=plt.subplots()
ax.scatter(C_test,new1)
ax.plot([C_test.min(),C_test.max()],[new1.min(),new1.max()],'k--',lw=1)
ax.set_xlabel('Measured Ultimate Yield Strength')
ax.set_ylabel('Predicted Ultimate Yield Strength')
plt.show()
  
#xlist = np.linspace(-3.0, 3.0, 100)
#ylist = np.linspace(-3.0, 3.0, 100)

X, Y = np.meshgrid(Mn_test, TE_test)
print(X.shape)
X=X.astype('int32')
Y=Y.astype('int32')
Z=np.sqrt(X**2+Y**2)
#Z = []
#for i in range(Mn_test.shape[0]):
    #Z.append((X[i]**2+Y[i]**2))
#Z=np.array(Z)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y,Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()
