import numpy as np
import ssl
import gbssl
import pandas as pd
from os import listdir
import time
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score as mutual_info

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"],
    "font.size": 15})

#Locale settings
#import locale
# Set to German locale to get comma decimal separater
#locale.setlocale(locale.LC_NUMERIC, "pt_BR.UTF-8")
#plt.rcParams['axes.formatter.use_locale'] = True

fig, ax = plt.subplots(2,2,sharex="col",figsize=(8,6))

# ------------------------------------------------------------------------------
# Import data
# ------------------------------------------------------------------------------

str_data_sparse = input("Similarity matrix (RBF - Sparse): ")

str_labels = input("Ground truth: ")
str_splits_1 = input("Splits file (0.02): ")
str_splits_2 = input("Splits file (0.10): ")
str_splits_3 = input("Splits file (0.20): ")
str_title = input("Dataset name: ")

#Load similarity matrix and ground truth
W = np.loadtxt(str_data_sparse)
labels = ssl.set_int_labels(np.loadtxt(str_labels))

#Number of variables and number of classes
N = len(labels); q = max(labels)+1
#Initialize model
m = gbssl.gmodel(W,q)

beta_range = 10**np.arange(-3,3,0.2)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# rl = 0.02
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

splits = np.loadtxt(str_splits_1)

#Accuracy rate
acc_rate = [[],[],[]]
#info Index
m_info = [[],[],[]]
#Execution time
exec_time = [[],[],[]]
#Log-probability
log_p = [[],[],[]]

#Iterate over beta_range
for k in range(len(beta_range)):
    #Auxiliary variables
    err = []
    info = []
    e_time = []
    aux_log_p = []
    #Iterate over splits
    for s in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[s]))
        #Calculate probabilities and store time
        start = time.time()
        m.nmf_propagation(beta_range[k],10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store info index
        info.append(mutual_info(labels,m.get_mode()))
        #Store log-probability
        aux_log_p.append(m.log_prob_mode()/N)
        #Reset probabilities
        m.reset_beliefs()
    #Store results for accuracy
    acc_rate[0].append(1.-np.mean(err))
    acc_rate[1].append(max(err)-np.mean(err))
    acc_rate[2].append(np.mean(err)-min(err))
    #Store results for info index
    m_info[0].append(np.mean(info))
    m_info[1].append(np.mean(info)-min(info))
    m_info[2].append(max(info)-np.mean(info))
    #Store results for execution time
    exec_time[0].append(np.mean(e_time))
    exec_time[1].append(np.mean(e_time)-min(e_time))
    exec_time[2].append(max(e_time)-np.mean(e_time))
    #Store results for log-probability
    log_p[0].append(np.mean(aux_log_p))
    log_p[1].append(np.mean(aux_log_p)-min(aux_log_p))
    log_p[2].append(max(aux_log_p)-np.mean(aux_log_p))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

ax[0,0].errorbar(beta_range,acc_rate[0],acc_rate[1:],marker='^',linestyle='solid',
    label=r'$r_{l}=0.02$',markevery=5,errorevery=5)
ax[0,1].errorbar(beta_range,m_info[0],m_info[1:],marker='^',linestyle='solid',
    label=r'$r_{l}=0.02$',markevery=5,errorevery=5)
ax[1,0].errorbar(beta_range,exec_time[0],exec_time[1:],marker='^',
    linestyle='solid',label=r'$r_{l}=0.02$',markevery=5,errorevery=5)
ax[1,1].errorbar(beta_range,log_p[0],log_p[1:],marker='^',
    linestyle='solid',label=r'$r_{l}=0.02$',markevery=5,errorevery=5)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# rl = 0.1
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

splits = np.loadtxt(str_splits_2)

#Accuracy rate
acc_rate = [[],[],[]]
#info Index
m_info = [[],[],[]]
#Execution time
exec_time = [[],[],[]]
#Log-probability
log_p = [[],[],[]]

#Iterate over beta_range
for k in range(len(beta_range)):
    #Auxiliary variables
    err = []
    info = []
    e_time = []
    aux_log_p = []
    #Iterate over splits
    for s in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[s]))
        #Calculate probabilities and store time
        start = time.time()
        m.nmf_propagation(beta_range[k],10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store info index
        info.append(mutual_info(labels,m.get_mode()))
        #Store log-probability
        aux_log_p.append(m.log_prob_mode()/N)
        #Reset probabilities
        m.reset_beliefs()
    #Store results for accuracy
    acc_rate[0].append(1.-np.mean(err))
    acc_rate[1].append(max(err)-np.mean(err))
    acc_rate[2].append(np.mean(err)-min(err))
    #Store results for info index
    m_info[0].append(np.mean(info))
    m_info[1].append(np.mean(info)-min(info))
    m_info[2].append(max(info)-np.mean(info))
    #Store results for execution time
    exec_time[0].append(np.mean(e_time))
    exec_time[1].append(np.mean(e_time)-min(e_time))
    exec_time[2].append(max(e_time)-np.mean(e_time))
    #Store results for log-probability
    log_p[0].append(np.mean(aux_log_p))
    log_p[1].append(np.mean(aux_log_p)-min(aux_log_p))
    log_p[2].append(max(aux_log_p)-np.mean(aux_log_p))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

ax[0,0].errorbar(beta_range,acc_rate[0],acc_rate[1:],marker='o',linestyle='dotted',
    label=r'$r_{l}=0.10$',markevery=5,errorevery=5)
ax[0,1].errorbar(beta_range,m_info[0],m_info[1:],marker='o',linestyle='dotted',
    label=r'$r_{l}=0.10$',markevery=5,errorevery=5)
ax[1,0].errorbar(beta_range,exec_time[0],exec_time[1:],marker='o',
    linestyle='dotted',label=r'$r_{l}=0.10$',markevery=5,errorevery=5)
ax[1,1].errorbar(beta_range,log_p[0],log_p[1:],marker='o',
    linestyle='dotted',label=r'$r_{l}=0.10$',markevery=5,errorevery=5)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# rl = 0.2
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

splits = np.loadtxt(str_splits_3)

#Accuracy rate
acc_rate = [[],[],[]]
#info Index
m_info = [[],[],[]]
#Execution time
exec_time = [[],[],[]]
#susceptibility
log_p = [[],[],[]]

#Iterate over beta_range
for k in range(len(beta_range)):
    #Auxiliary variables
    err = []
    info = []
    e_time = []
    aux_log_p = []
    #Iterate over splits
    for s in range(len(splits)):
        m.set_fields(ssl.get_fields(labels,splits[s]))
        #Calculate probabilities and store time
        start = time.time()
        m.nmf_propagation(beta_range[k],10000,0.001)
        e_time.append(time.time()-start)
        #Store error
        err.append(ssl.get_error(m.get_mode(),labels))
        #Store info index
        info.append(mutual_info(labels,m.get_mode()))
        #Store log-probability
        aux_log_p.append(m.log_prob_mode()/N)
        #Reset probabilities
        m.reset_beliefs()
    #Store results for accuracy
    acc_rate[0].append(1.-np.mean(err))
    acc_rate[1].append(max(err)-np.mean(err))
    acc_rate[2].append(np.mean(err)-min(err))
    #Store results for info index
    m_info[0].append(np.mean(info))
    m_info[1].append(np.mean(info)-min(info))
    m_info[2].append(max(info)-np.mean(info))
    #Store results for execution time
    exec_time[0].append(np.mean(e_time))
    exec_time[1].append(np.mean(e_time)-min(e_time))
    exec_time[2].append(max(e_time)-np.mean(e_time))
    #Store results for log-probability
    log_p[0].append(np.mean(aux_log_p))
    log_p[1].append(np.mean(aux_log_p)-min(aux_log_p))
    log_p[2].append(max(aux_log_p)-np.mean(aux_log_p))
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

ax[0,0].errorbar(beta_range,acc_rate[0],acc_rate[1:],marker='v',linestyle='dashed',
    label=r'$r_{l}=0.20$)',markevery=5,errorevery=5)
ax[0,1].errorbar(beta_range,m_info[0],m_info[1:],marker='v',linestyle='dashed',
    label=r'$r_{l}=0.20$)',markevery=5,errorevery=5)
ax[1,0].errorbar(beta_range,exec_time[0],exec_time[1:],marker='v',
    linestyle='dashed',label=r'$r_{l}=0.20$',markevery=5,errorevery=5)
ax[1,1].errorbar(beta_range,log_p[0],log_p[1:],marker='v',
    linestyle='dashed',label=r'$r_{l}=0.20$',markevery=5,errorevery=5)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting - Final
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

ax[0,0].set_xlabel(r'$\beta$')
ax[0,1].set_xlabel(r'$\beta$')
ax[1,0].set_xlabel(r'$\beta$')
ax[1,1].set_xlabel(r'$\beta$')

ax[0,0].set_ylabel('Accuracy')
ax[0,1].set_ylabel('AMI (nats)')
ax[1,0].set_ylabel('Time (s)')
ax[1,1].set_ylabel(r'$\frac{1}{N}\ln \Gamma$')

ax[0,0].set_xscale('log')
ax[0,1].set_xscale('log')
ax[1,0].set_xscale('log')
ax[1,1].set_xscale('log')

ax[1,0].set_yscale('log')

ax[0,0].grid(True)
ax[0,1].grid(True)
ax[1,0].grid(True)
ax[1,1].grid(True)

ax[0,0].legend(ncol=3,bbox_to_anchor=(1.1, 1.22),loc='upper center')

l1 = ax[1,1].axhline(np.log(1/q),linestyle='dashdot',color='black',
    label = r'$\Gamma = \big(\frac{1}{q}\big)^{N}$')
l2 = ax[1,1].axhline(np.log((1+1/q)/2),linestyle='dashdot',color='gray',
    label = r'$\Gamma = \big(\frac{1+q}{2q}\big)^{N}$')
ax[1,1].legend(handles=[l1,l2])

plt.suptitle(str_title+r" ($k=\log_{2}N$)")

plt.show()
