import numpy as np

pi = 3.141592684
#Number of instances
N = 900

data = [[],[]]
labels = []

#Generate dataset
for i in range(N):
    p = np.random.uniform(0.,1.)
    theta = np.random.uniform(0.,np.pi)
    if p<1/3:
        data[0].append(np.random.uniform(-0.5,0.5))
        data[1].append(np.random.uniform(-0.5,0.5))
        labels.append(0)
    else:
        if p<2/3:
            r = 2.+np.random.normal(scale=0.2)
            data[0].append(r*np.cos(theta))
            data[1].append(r*np.sin(theta))
            labels.append(1)
        else:
            r = 3+np.random.normal(scale=0.1)
            data[0].append(r*np.cos(theta))
            data[1].append(-r*np.sin(theta))
            labels.append(2)


#Save dataset
np.savetxt("circles_data.dat",data)
#Save ground truth
np.savetxt("circles_labels.dat",labels)

import matplotlib.pyplot as plt

#Plot dataset
for i in range(N):
    if labels[i] == 0:
        plt.plot(data[0][i],data[1][i],markerfacecolor="none",
        markeredgecolor='b',marker='o',linewidth=0.)
    if labels[i] == 1:
        plt.plot(data[0][i],data[1][i],markerfacecolor="none",
        markeredgecolor='r', marker='s',linewidth=0.)
    if labels[i] == 2:
        plt.plot(data[0][i],data[1][i],markerfacecolor="none",
        markeredgecolor='g', marker='v',linewidth=0.)
plt.axis('off')
plt.show()

#Function to generate splits
def get_splits(labels,n_splits,n_labeled):
    """Returns n_splits with n_labeled data points."""
    N = len(labels); q  = max(labels)+1
    splits = []
    n = 0
    while(n<n_splits):
        s = []
        k = 0
        label_count = [0 for i in range(q)]
        #Generate a random split
        while(k<n_labeled):
            m = np.random.randint(N)
            if m not in s:
                s.append(m)
                label_count[labels[m]] += 1
                k += 1
        #Check if every class has an element in s
        if (0 not in label_count):
            splits.append(s)
            n += 1
    return splits

splits = get_splits(labels,50,int(N*0.1))
#Write splits to file
np.savetxt("circles_splits.dat",splits)

#Square distance matrix
sqr_d = np.zeros((N,N))
for i in range(N):
    for j in range(i+1,N):
        sqr_d[i][j] = (data[0][i]-data[0][j])**2+(data[1][i]-data[1][j])**2
        sqr_d[j][i] = sqr_d[i][j]

#Save square distance
np.savetxt("circles_sqr_dist.dat",sqr_d)
