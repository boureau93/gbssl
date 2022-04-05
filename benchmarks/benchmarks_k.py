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

str_data = input("Similarity folder: ")
str_labels = input("Ground truth: ")
str_splits = input("Splits: ")
str_title = input("Dataset name: ")

#Load ground truth
labels = ssl.set_int_labels(np.loadtxt(str_labels))

#Splits
splits = np.loadtxt(str_splits)

#Number of variables and number of classes
N = len(labels); q = max(labels)+1
r = len(splits[0])/N
#Load similarity matrixes
W_files = sorted(listdir(str_data))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

#Accuracy rate
acc_rate_grf = [[],[],[]]
acc_rate_lgc = [[],[],[]]
acc_rate_nmf_A = [[],[],[]]
acc_rate_nmf_B = [[],[],[]]
#Adjusted mutual info
ami_grf = [[],[],[]]
ami_lgc = [[],[],[]]
ami_nmf_A = [[],[],[]]
ami_nmf_B = [[],[],[]]
#Execution time
exec_time_grf = [[],[],[]]
exec_time_lgc = [[],[],[]]
exec_time_nmf_A = [[],[],[]]
exec_time_nmf_B = [[],[],[]]

#Iterate over similarities
for s in W_files:
    print(s)
    #Load similarities and normalize
    W = np.loadtxt(str_data+'/'+s)
    #Initialize model
    m = gbssl.gmodel(W,q)

    
    #Auxiliary variables to store results
    err = []
    info = []
    e_time = []
    #Iterate over splits (LP)
    print("GRF")
    for k in range(len(splits)):
        #Set labeled data
        m.set_fields(ssl.get_fields(labels,splits[k]))
        #Iterate GRF
        start = time.time()
        m.grf(10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store mutual info
        info.append(mutual_info(labels,m.get_mode()))
        #Reset probabilities
        m.reset_beliefs()
    #Write accuracy results to arrays
    acc_rate_grf[0].append(1.-np.mean(err))
    acc_rate_grf[1].append(max(err)-np.mean(err))
    acc_rate_grf[2].append(np.mean(err)-min(err))
    #Write mutual information results to arrays
    ami_grf[0].append(np.mean(info))
    ami_grf[1].append(np.mean(info)-min(info))
    ami_grf[2].append(max(info)-np.mean(info))
    #Write time results to arrays
    exec_time_grf[0].append(np.mean(e_time))
    exec_time_grf[1].append(np.mean(e_time)-min(e_time))
    exec_time_grf[2].append(max(e_time)-np.mean(e_time))
    
    #Auxiliary variables for error rate and execution time
    err = []
    info = []
    e_time = []
    #Iterate over splits (LGC)
    print("LGC")
    for k in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[k]))
        #Iterate LGC
        start = time.time()
        m.lgc(0.99,10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store mutual info
        info.append(mutual_info(labels,m.get_mode())) 
        #Reset probabilities
        m.reset_beliefs()
    #Write accuracy results to arrays
    acc_rate_lgc[0].append(1.-np.mean(err))
    acc_rate_lgc[1].append(max(err)-np.mean(err))
    acc_rate_lgc[2].append(np.mean(err)-min(err))
    #Write mutual information results to arrays
    ami_lgc[0].append(np.mean(info))
    ami_lgc[1].append(np.mean(info)-min(info))
    ami_lgc[2].append(max(info)-np.mean(info))
    #Write time results to arrays
    exec_time_lgc[0].append(np.mean(e_time))
    exec_time_lgc[1].append(np.mean(e_time)-min(e_time))
    exec_time_lgc[2].append(max(e_time)-np.mean(e_time))
    
    #Auxiliary variables
    err = []
    info = []
    e_time = []
    #Iterate over splits (NMF - beta_max)
    print("NMF A")
    for k in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[k]))
        #Iterate NMF
        start = time.time()
        m.nmf_propagation(get_beta(r,q,(1.+q)/(2.*q)),10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store mutual information
        info.append(mutual_info(labels,m.get_mode()))
        #Reset probabilities
        m.reset_beliefs()
    #Write accuracy results to arrays
    acc_rate_nmf_A[0].append(1.-np.mean(err))
    acc_rate_nmf_A[1].append(max(err)-np.mean(err))
    acc_rate_nmf_A[2].append(np.mean(err)-min(err))
    #Write mutual information results to arrays
    ami_nmf_A[0].append(np.mean(info))
    ami_nmf_A[1].append(np.mean(info)-min(info))
    ami_nmf_A[2].append(max(info)-np.mean(info))
    #Write time results to arrays
    exec_time_nmf_A[0].append(np.mean(e_time))
    exec_time_nmf_A[1].append(np.mean(e_time)-min(e_time))
    exec_time_nmf_A[2].append(max(e_time)-np.mean(e_time))
    
    #Auxiliary variables 
    err = []
    info = []
    e_time = []
    #Iterate over splits (NMF - beta_avg)
    print("NMF B")
    for k in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[k]))
        #Iterate NMF
        start = time.time()
        m.nmf_propagation(get_beta(r,q,1.),10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store mutual info
        info.append(mutual_info(labels,m.get_mode()))
        #Reset probabilities
        m.reset_beliefs()
    #Write accuracy results to arrays
    acc_rate_nmf_B[0].append(1.-np.mean(err))
    acc_rate_nmf_B[1].append(max(err)-np.mean(err))
    acc_rate_nmf_B[2].append(np.mean(err)-min(err))
    #Write mutual information results to arrays
    ami_nmf_B[0].append(np.mean(info))
    ami_nmf_B[1].append(np.mean(info)-min(info))
    ami_nmf_B[2].append(max(info)-np.mean(info))
    #Write time results to arrays
    exec_time_nmf_B[0].append(np.mean(e_time))
    exec_time_nmf_B[1].append(np.mean(e_time)-min(e_time))
    exec_time_nmf_B[2].append(max(e_time)-np.mean(e_time))
    
#Write results to file
np.savetxt(str_title+'_grf_acc.dat',acc_rate_grf)
np.savetxt(str_title+'_grf_ami.dat',ami_grf)
np.savetxt(str_title+'_grf_time.dat',exec_time_grf)
np.savetxt(str_title+'_lgc_acc.dat',acc_rate_lgc)
np.savetxt(str_title+'_lgc_ami.dat',ami_lgc)
np.savetxt(str_title+'_lgc_time.dat',exec_time_lgc)
np.savetxt(str_title+'_nmfA_acc.dat',acc_rate_nmf_A)
np.savetxt(str_title+'_nmfA_ami.dat',ami_nmf_A)
np.savetxt(str_title+'_nmfA_time.dat',exec_time_nmf_A)
np.savetxt(str_title+'_nmfB_acc.dat',acc_rate_nmf_B)
np.savetxt(str_title+'_nmfB_ami.dat',ami_grf)
np.savetxt(str_title+'_nmfB_time.dat',exec_time_nmf_B)
