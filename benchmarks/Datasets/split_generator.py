
import numpy as np

def get_splits(labels,n_splits,n_labeled):
    """Returns n_splits with n_labeled data points."""
    N = len(labels); q  = int(max(labels)+1)
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
                label_count[int(labels[m])] += 1
                k += 1
        #Check if every class has an element in s
        if (0 not in label_count):
            splits.append(s)
            n += 1
    return splits

#Ground truth
str_labels = input("Ground truth: ")
labels = np.loadtxt(str_labels)
N = len(labels)
for i in range(N):
    labels[i] = int(labels[i])

#Number of splits
n_splits = eval(input("Number of splits to generate: "))

for p in range(0,10):
    splits = get_splits(labels,n_splits,int((p+1)*2*N/100))
    np.savetxt("splits_"+str(p)+".dat",splits)
