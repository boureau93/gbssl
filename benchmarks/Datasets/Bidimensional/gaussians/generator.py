import numpy as np

#Number of instances
N = 1000
labels = []
#1 gaussian
data = [[],[]]
data[0] = np.random.normal(1,0.7,int(2*N/5))
data[1] = np.random.normal(1,0.7,int(2*N/5))
for i in range(int(2*N/5)):
    labels.append(0)

#2 gaussian
data[0] = np.concatenate((data[0],np.random.normal(-1,0.2,int(N/5))))
data[1] = np.concatenate((data[1],np.random.normal(-1,0.3,int(N/5))))
for i in range(int(N/5)):
    labels.append(1)

#3 gaussian
data[0] = np.concatenate((data[0],np.random.normal(-1,0.2,int(N/5))))
data[1] = np.concatenate((data[1],np.random.normal(1,0.1,int(N/5))))
for i in range(int(N/5)):
    labels.append(2)

#4 gaussian
data[0] = np.concatenate((data[0],np.random.normal(1,0.1,int(N/10))))
data[1] = np.concatenate((data[1],np.random.normal(-1,0.1,int(N/10))))
for i in range(int(N/10)):
    labels.append(3)

#5 gaussian
data[0] = np.concatenate((data[0],np.random.normal(1,0.1,int(N/10))))
data[1] = np.concatenate((data[1],np.random.normal(-2,0.1,int(N/10))))
for i in range(int(N/10)):
    labels.append(4)

#Write data to file
np.savetxt("gaussians_data.dat",data)

#Plot dataset and set labels
import matplotlib.pyplot as plt
for i in range(N):
    if labels[i]==0:
        plt.plot(data[0][i],data[1][i],marker='o',markerfacecolor='none',
            markeredgecolor='b')
    if labels[i]==1:
        plt.plot(data[0][i],data[1][i],marker='s',markerfacecolor='none',
            markeredgecolor='r')
    if labels[i]==2:
        plt.plot(data[0][i],data[1][i],marker='v',markerfacecolor='none',
            markeredgecolor='g')
    if labels[i]==3:
        plt.plot(data[0][i],data[1][i],marker='p',markerfacecolor='none',
            markeredgecolor='y')
    if labels[i]==4:
        plt.plot(data[0][i],data[1][i],marker='^',markerfacecolor='none',
            markeredgecolor='m')
plt.axis('off')
plt.show()

#Write labels to file
np.savetxt("gaussians_labels.dat",labels)

#Square distance matrix
sqr_d = np.zeros((N,N))
for i in range(N):
    for j in range(i+1,N):
        sqr_d[i][j] = (data[0][i]-data[0][j])**2+(data[1][i]-data[1][j])**2
        sqr_d[j][i] = sqr_d[i][j]

#Write sqr_d to file
np.savetxt("gaussians_sqr_dist.dat",sqr_d)
