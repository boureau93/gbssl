import numpy as np

# ------------------------------------------------------------------------------
# Import data
# ------------------------------------------------------------------------------

str_data = input("Square distance matrix: ")
str_title = input("Dataset name: ")

#Load distance matrix and ground truth
dist = np.sqrt(np.loadtxt(str_data))

#Number of variables
N = len(dist)

# ------------------------------------------------------------------------------
# Similarity construction
# ------------------------------------------------------------------------------

#Range of sparsification parameter
k_range = [k for k in range(2,21)]

for k in k_range:
    #set kNN similarity matrix and calculate scale parameter
    sigma = 0.
    J = np.zeros((N,N))
    for i in range(N):
        aux = np.sort(dist[i])
        sigma += aux[k]
    sigma /= 3*N
    #Calculate J
    for i in range(N):
    	aux = np.sort(dist[i])
    	for j in range(1,k+1):
    		p = np.where(dist[i]==aux[j])
    		J[i][p] = np.exp(-dist[i][p]*dist[i][p]/(2*sigma*sigma))
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
    #Save to file
    if (k<12):
        np.savetxt(str_title+'_0'+str(k-2)+'.dat',W)
    else:
        np.savetxt(str_title+'_1'+str(k-12)+'.dat',W)
