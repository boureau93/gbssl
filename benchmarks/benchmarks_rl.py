import numpy as np
import ssl
import gbssl
import pandas as pd
from os import listdir
import time
from sklearn.metrics import adjusted_mutual_info_score as mutual_info
from scipy.optimize import root_scalar as rs

#-------------------------------------------------------------------------------
#Root finding for beta
#-------------------------------------------------------------------------------
def f(x,r,q,alpha):
    return x*(r+1./q)-r*np.log(q+2*x)-(1-r)*np.log(q+x)-np.log(alpha)

def f_prime(x,r,q,alpha):
    return r+1./q-r*(2/(q+2*x))-(1-r)/(q+x)

def get_beta(r,q,alpha):
    beta = rs(f,(r,q,alpha),x0=1.,xtol=0.001,fprime=f_prime,method='newton').root 
    return beta 

# ------------------------------------------------------------------------------
# Import data
# ------------------------------------------------------------------------------

str_data = input("Similarity matrix: ")
str_labels = input("Ground truth: ")
str_splits = input("Splits folder: ")
str_title = input("Dataset name: ")

#Load similarity matrix and ground truth
W = np.loadtxt(str_data)
labels = ssl.set_int_labels(np.loadtxt(str_labels))

#Splits files
splits_files = sorted(listdir(str_splits))

#Number of variables and number of classes
N = len(labels); q = max(labels)+1
#Initialize model
m = gbssl.gmodel(W,q)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# GRF
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

print("GRF")

#Rate of labeled points
label_rate = []
#Accuracy rate
acc_rate = [[],[],[]]
#Execution time
exec_time = [[],[],[]]
#Adjusted mutual information
ami = [[],[],[]]

#Iterate over split files
for str_s in splits_files:
    print(str_s)
    #Load splits and cast elements to int
    aux_splits = np.loadtxt(str_splits+"/"+str_s)
    splits = []
    for k in range(len(aux_splits)):
        splits.append([])
        for s in aux_splits[k]:
            splits[k].append(int(s))

    #Store rate of labeled points
    label_rate.append(len(splits[0])/N)

    #Auxiliary variables
    err = []
    e_time = []
    info = []
    #Iterate over splits
    for s in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[s]))
        #Calculate probabilities and store time
        start = time.time()
        m.grf(10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store mutual information
        info.append(mutual_info(labels,m.get_mode()))
        #Reset probabilities
        m.reset_beliefs()

    acc_rate[0].append(1.-np.mean(err))
    acc_rate[1].append(max(err)-np.mean(err))
    acc_rate[2].append(np.mean(err)-min(err))

    ami[0].append(np.mean(info))
    ami[1].append(np.mean(info)-min(info))
    ami[2].append(max(info)-np.mean(info))

    exec_time[0].append(np.mean(e_time))
    exec_time[1].append(np.mean(e_time)-min(e_time))
    exec_time[2].append(max(e_time)-np.mean(e_time))

#Write results to file
np.savetxt(str_title+'_grf_acc.dat',acc_rate)
np.savetxt(str_title+'_grf_ami.dat',ami)
np.savetxt(str_title+'_grf_time.dat',exec_time)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Local and Global Consistency
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

print("LGC")

#Rate of labeled points
label_rate = []
#Accuracy rate
acc_rate = [[],[],[]]
#Execution time
exec_time = [[],[],[]]
#Adjusted mutual information
ami = [[],[],[]]

#Iterate over split files
for str_s in splits_files:
    print(str_s)
    #Load splits and cast elements to int
    aux_splits = np.loadtxt(str_splits+"/"+str_s)
    splits = []
    for k in range(len(aux_splits)):
        splits.append([])
        for s in aux_splits[k]:
            splits[k].append(int(s))

    #Store rate of labeled points
    label_rate.append(len(splits[0])/N)

    #Auxiliary variable for error rate and time
    err = []
    e_time = []
    info = []
    #Iterate over splits
    for s in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[s]))
        #Calculate probabilities and store time
        start = time.time()
        t = m.lgc(0.99,10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store mutual information
        info.append(mutual_info(labels,m.get_mode()))
        #Reset probabilities
        m.reset_beliefs()

    acc_rate[0].append(1.-np.mean(err))
    acc_rate[1].append(max(err)-np.mean(err))
    acc_rate[2].append(np.mean(err)-min(err))

    ami[0].append(np.mean(info))
    ami[1].append(np.mean(info)-min(info))
    ami[2].append(max(info)-np.mean(info))

    exec_time[0].append(np.mean(e_time))
    exec_time[1].append(np.mean(e_time)-min(e_time))
    exec_time[2].append(max(e_time)-np.mean(e_time))

#Write results to file
np.savetxt(str_title+'_lgc_acc.dat',acc_rate)
np.savetxt(str_title+'_lgc_ami.dat',ami)
np.savetxt(str_title+'_lgc_time.dat',exec_time)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Naive mean fields
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

print("NMF A")

#Rate of labeled points
label_rate = []
#Accuracy rate
acc_rate = [[],[],[]]
#Execution time
exec_time = [[],[],[]]
#Adjusted mutual information
ami = [[],[],[]]

#Iterate over split files
for str_s in splits_files:
    print(str_s)
    #Load splits and cast elements to int
    aux_splits = np.loadtxt(str_splits+"/"+str_s)
    splits = []
    for k in range(len(aux_splits)):
        splits.append([])
        for s in aux_splits[k]:
            splits[k].append(int(s))

    #Store rate of labeled points
    label_rate.append(len(splits[0])/N)

    #Auxiliary variable for error, time and magnetization
    err = []
    e_time = []
    info = []
    #Iterate over splits
    for s in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[s]))
        #Calculate probabilities and store time
        start = time.time()
        m.nmf_propagation(get_beta(label_rate[-1],q,(1.+q)/(2.*q)),10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store mutual information
        info.append(mutual_info(labels,m.get_mode()))
        #Reset probabilities
        m.reset_beliefs()

    acc_rate[0].append(1.-np.mean(err))
    acc_rate[1].append(max(err)-np.mean(err))
    acc_rate[2].append(np.mean(err)-min(err))

    ami[0].append(np.mean(info))
    ami[1].append(np.mean(info)-min(info))
    ami[2].append(max(info)-np.mean(info))

    exec_time[0].append(np.mean(e_time))
    exec_time[1].append(np.mean(e_time)-min(e_time))
    exec_time[2].append(max(e_time)-np.mean(e_time))

#Write results to file
np.savetxt(str_title+'_nmfA_acc.dat',acc_rate)
np.savetxt(str_title+'_nmfA_ami.dat',ami)
np.savetxt(str_title+'_nmfA_time.dat',exec_time)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Naive mean fields
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

print("NMF B")

#Rate of labeled points
label_rate = []
#Accuracy rate
acc_rate = [[],[],[]]
#Execution time
exec_time = [[],[],[]]
#Adjusted mutual information
ami = [[],[],[]]

#Iterate over split files
for str_s in splits_files:
    print(str_s)
    #Load splits and cast elements to int
    aux_splits = np.loadtxt(str_splits+"/"+str_s)
    splits = []
    for k in range(len(aux_splits)):
        splits.append([])
        for s in aux_splits[k]:
            splits[k].append(int(s))

    #Store rate of labeled points
    label_rate.append(len(splits[0])/N)

    #Auxiliary variable for error, time and magnetization
    err = []
    e_time = []
    info = []
    #Iterate over splits
    for s in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[s]))
        #Calculate probabilities and store time
        start = time.time()
        m.nmf_propagation(get_beta(label_rate[-1],q,1.),10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store mutual information
        info.append(mutual_info(labels,m.get_mode()))
        #Reset probabilities
        m.reset_beliefs()

    acc_rate[0].append(1.-np.mean(err))
    acc_rate[1].append(max(err)-np.mean(err))
    acc_rate[2].append(np.mean(err)-min(err))

    ami[0].append(np.mean(info))
    ami[1].append(np.mean(info)-min(info))
    ami[2].append(max(info)-np.mean(info))

    exec_time[0].append(np.mean(e_time))
    exec_time[1].append(np.mean(e_time)-min(e_time))
    exec_time[2].append(max(e_time)-np.mean(e_time))

#Write results to file
np.savetxt(str_title+'_nmfB_acc.dat',acc_rate)
np.savetxt(str_title+'_nmfB_ami.dat',ami)
np.savetxt(str_title+'_nmfB_time.dat',exec_time)
