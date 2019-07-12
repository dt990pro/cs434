import numpy as np
import sys
import csv
import math

def normal(X):
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.min(X[:,i]))/(np.max(X[:,i]) - np.min(X[:,i]))

    return X

def change_to_zero(X):
    for i in range(X.shape[0]):
        if (X[i] == -1):
            X[i] = 0

    return X

def read(str):
    with open(str) as csvfile:
        readCSV = csv.reader(csvfile)
        results=[]
        features=[]
        for row in readCSV:
            result=row[0]	    #result of each row
            feature=row[1:]     #feature of each row

            features.append(feature) 
            results.append(result)

    x = np.array(features, dtype = float)
    y = np.array(results, dtype = float)
    return x, y

# entropy
def H(s):
    pos = (s[:, 0] == 1).sum() / s.shape[0]
    neg = 1 - pos
    if (pos == 0 or neg == 0):
        return 0
    else:
        entropy = - pos * math.log2(pos) - neg * math.log2(neg)
        return entropy

def compute_gain(i, set):
    s1 = set[ set[:, 1] < i ]
    s2 = set[ set[:, 1] >= i ]
    p1 = s1.shape[0] / set.shape[0]
    p2 = s2.shape[0] / set.shape[0]
    entropy1 = 0
    entropy2 = 0
    if (s1.shape[0] != 0):
        entropy1 = p1 * H(s1)
    if (s2.shape[0] != 0):
        entropy2 = p2 * H(s2)
    # compute by entropy
    gain = H(set) - entropy1 - entropy2
    return gain

def choosing_this_threshold(feature, test):
    test = test.reshape(test.shape[0], 1)
    feature = feature.reshape(feature.shape[0], 1)
    set = np.concatenate((test, feature), axis=1)
    # sort feature column
    set = set[set[:, 1].argsort()]

    # form a list of thresholds
    thresholds = []
    for i in range(set.shape[0] - 1):
        thresholds.append( (set[i][1] + set[i+1][1]) / 2. )

    # find the best threshold
    info_gain = -float('inf')
    for i in thresholds:
        temp_gain = compute_gain(i, set)
        if (temp_gain > info_gain):
            info_gain = temp_gain
            threshold = i

    return threshold, info_gain

def choosing_the_best(train, test):
    info_gain = -float('inf')
    # compute each feature 0 - 29
    for i in range(train.shape[1]):
        # compute feature i's threashold, gain
        i_threshold, i_gain = choosing_this_threshold(train[:, i], test)

        # update
        if (i_gain > info_gain):
            info_gain = i_gain
            feature = i
            threshold = i_threshold

    return threshold, feature, info_gain

def split(train, test, threshold, feature):
    s1 = train[ train[:, feature] < threshold ]
    s2 = train[ train[:, feature] >= threshold ]
    t1 = test[ train[:, feature] < threshold ]
    t2 = test[ train[:, feature] >= threshold ]
    return s1, s2, t1, t2

class decision_stump(object):
    def __init__(self, train, test, depth):
        self.threshold, self.feature, self.info_gain = choosing_the_best(train, test)
        s1, s2, t1, t2 = split(train, test, self.threshold, self.feature)
        t1 = t1.reshape(t1.shape[0], 1)
        t2 = t2.reshape(t2.shape[0], 1)

        left_pos = (t1[:, 0] == 1).sum()
        left_neg = (t1[:, 0] == -1).sum()
        right_pos = (t2[:, 0] == 1).sum()
        right_neg = (t2[:, 0] == -1).sum()

        if (depth == 1):
            self.left_node = ('+:' + str(left_pos), '-:' + str(left_neg), left_pos / (left_pos + left_neg))
            self.right_node = ('+:' + str(right_pos), '-:' + str(right_neg), right_pos / (right_pos + right_neg))
        else:
            # check left node
            if (t2.shape[0] == 0 or len(np.unique(t1[:, 0])) == 1):     # no right or has same label
                self.left_node = ('+:' + str(left_pos), '-:' + str(left_neg))
            else:
                self.left_node = decision_stump(s1, t1, depth - 1)
            # check right node
            if (t1.shape[0] == 0 or len(np.unique(t2[:, 0])) == 1):     # no left or has same lable
                self.right_node = ('+:' + str(right_pos), '-:' + str(right_neg))
            else:
                self.right_node = decision_stump(s2, t2, depth - 1)

def print_tree(dec_stump, depth):
    if (depth == 1):
        print('feature', dec_stump.feature + 1)     # since arr starts at 0
        print('info_gain', dec_stump.info_gain)
        print(dec_stump.left_node, '<to left----(<)', dec_stump.threshold, '(>=)----to right>', dec_stump.right_node)

    return

def get_label(node, x):
    # to left node
    if (x[node.feature] < node.threshold):
        # is tuple, get label
        if isinstance(node.left_node, tuple):
            return node.left_node[2]    # label in third position of tuple
        # expand left
        else:
            return get_label(node.left_node, x)
    # to right node
    else:
        # is tuple, get label
        if isinstance(node.right_node, tuple):
            return node.right_node[2]    # label in third position of tuple
        # expand right
        else:
            return get_label(node.right_node, x)

def err_for(node, x, y):
    err = 0
    # check each case
    for i in range(x.shape[0]):
        label = get_label(node, x[i])
        if (label < 0.5 and y[i] == 1):
            err += 1
        elif (label >= 0.5 and y[i] == -1):
            err += 1
        
    print(err / y.shape[0])
    return

train_x, train_y = read(sys.argv[1])
test_x, test_y = read(sys.argv[2])
depth = 1

# build
dec_stump = decision_stump(train_x, train_y, 1)
# print
print_tree(dec_stump, depth)
print('train err')
err_for(dec_stump, train_x, train_y)
print('test err')
err_for(dec_stump, test_x, test_y)