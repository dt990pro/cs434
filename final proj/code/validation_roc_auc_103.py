import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import re
import math
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

### read file
def read(path):
    feature = open(path, "r").readlines()

    train = []
    validation = []

    for i in range(len(feature)):
        if i == 0:
            continue

        if i % 5 == 0:
            # add validation
            str = re.split('[\n\t]', feature[i])[1:105]
            validation.append(str)
        else:
            str = re.split('[\n\t]', feature[i])[1:105]
            train.append(str)

    return np.array(train, dtype = float), np.array(validation, dtype = float)

def main():
    # read
    train, validation = read('feature103_Train.txt')
    train_x = train[:, 1:]
    train_y = train[:, 0]
    train_y = train_y.reshape(train_y.shape[0], 1)
    validation_x = validation[:, 1:]
    validation_y = validation[:, 0]
    validation_y = validation_y.reshape(validation_y.shape[0], 1)

    # normalize
    base = 10.
    
    train_x         = np.log(train_x + 10 ) / np.log(base)
    validation_x    = np.log(validation_x + 10) / np.log(base)
    train_x         = tf.keras.utils.normalize(train_x)
    validation_x    = tf.keras.utils.normalize(validation_x)
    
    # build up model
    model = load_model('model_103v2')	# change model here

    result = model.predict(validation_x)

    print(result)
    print(result.shape)

    for i in range(len(result)):
        if result[i] < 0.5:
            result[i] = 0.
        else:
            result[i] = 1.

    print(result)
    print(result.shape)

    roc_val = roc_auc_score(validation_y, result)
    print(roc_val)

    return 0


main()