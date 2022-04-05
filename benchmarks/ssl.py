import numpy as np

def set_int_labels(labels):
    """Set labels to integers."""
    label_map = []
    new_labels = []
    for x in labels:
        if x not in label_map:
            label_map.append(x)
        new_labels.append(label_map.index(x))
    return new_labels

def get_fields(labels,split):
    """Set a field matrix"""
    N = len(labels)
    q = max(labels)+1
    theta = np.zeros((N,q))
    for p in split:
        theta[int(p)][labels[int(p)]] = 1.
    return theta

# ------------------------------------------------------------------------------
# Benchmark functions
# ------------------------------------------------------------------------------

def get_error(ans,labels):
    """Returns the error between a classification (ans) and ground truth(labels)
    """
    if (len(ans)==len(labels)):
        err = 0.
        for i in range(len(ans)):
            if int(ans[i])!=int(labels[i]):
                err += 1.
        return err/len(ans)
    else:
        print("ans and labels have different shapes.")

def get_error_split(ans,labels,split):
    """Returns the error between a classification (ans) and ground truth(labels)
    excluding instances in split."""
    if (len(ans)==len(labels)):
        err = 0.
        for i in range(len(ans)):
            if i not in split:
                if int(ans[i])!=int(labels[i]):
                    err += 1.
        return err/(len(ans)-len(split))
    else:
        print("ans and labels have different shapes.")