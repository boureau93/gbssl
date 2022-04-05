import numpy as np

#Input files
str_dist = input("Distance matrix: ")
str_title = input("Dataset name: ")
sqr_dist = np.loadtxt(str_dist)

#Number of instances
N = len(sqr_dist)
#Number of nearest neighbors
k = int(np.log2(N))

#set kNN distance matrix and calculate scale parameter
sigma = 0.
for i in range(N):
    aux = np.sort(sqr_dist[i])
    sigma += np.sqrt(aux[k])
sigma /= 3*N

#Set similarity
J = np.zeros((N,N))
for i in range(N):
    aux = np.sort(sqr_dist[i])
    for j in range(1,k+1):
        p = np.where(sqr_dist[i]==aux[j])
        J[i][p] = np.exp(-sqr_dist[i][p]/(2*sigma*sigma))

#Set sparse similarity
W = np.zeros((N,N))
for i in range(N):
    aux = 0
    for j in range(N):
        W[i][j] = min([J[i][j],J[j][i]])
        if (j==np.argmax(J[i]) and W[i][j]==0.):
            W[i][j] = J[i][j]
W = (W+np.transpose(W))/2.
#Normalize similarity
D = np.zeros((N,N))
for i in range(N):
    D[i][i] = 1/np.sqrt(np.sum(W[i]))
W = np.matmul(D,np.matmul(W,D))

#Save J to file
np.savetxt(str_title+"_rbf_full.dat",W)
