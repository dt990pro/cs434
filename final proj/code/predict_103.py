import numpy as np
import sys
import re
import math
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

def read_test(path):
    file = open(path, "r").readlines()

    id = []
    test = []

    for i in range(len(file)):
        if i == 0:
            continue

        str = re.split('[\n\t]', file[i])[0:1]
        id.append(str)

        str = re.split('[\n\t]', file[i])[1:104]
        test.append(str)

    return id, np.array(test, dtype = float)

def predict():
    id, test = read_test('features103_test.txt')

    # normalize
    base = 10.
    
    test         = np.log(test + 10 ) / np.log(base)
    test         = tf.keras.utils.normalize(test)

    model = load_model('model_103v0')

    result = model.predict(test)

    print(id)
    print(result)
    print(result.shape)

    ### print
    f = open("features103_pred1.txt","w")
    for i in range(len(id)):
        print(str(id[i][0]) + ',' + str(result[i][0]), file = f)

    print('done')
    f.close()

    return 0

predict()