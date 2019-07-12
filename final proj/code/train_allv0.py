import numpy as np
import sys
import re
import math
import tensorflow as tf
import tensorflow.keras as keras
#from tensorflow.keras.callbacks import TensorBoard

#print(tf.__version__)



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
            str = re.split('[\n\t]', feature[i])[1:1055]
            validation.append(str)
        else:
            str = re.split('[\n\t]', feature[i])[1:1055]
            train.append(str)

    return np.array(train, dtype = float), np.array(validation, dtype = float)

### build up model
def get_model():
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense( 512, activation = tf.nn.relu ))
    model.add(tf.keras.layers.Dense( 1024, activation = tf.nn.relu ))
    model.add(tf.keras.layers.Dense( 1, activation = tf.nn.sigmoid ))

    #model.add(tf.keras.layers.Dense( 64, activation = tf.nn.relu ))
    #model.add(tf.keras.layers.Dense( 256, activation = tf.nn.relu ))
    #model.add(tf.keras.layers.Dense( 64, activation = tf.nn.sigmoid ))
    #model.add(tf.keras.layers.Dense( 1, activation = tf.nn.sigmoid ))
    

    return model

def main():
    # read
    train, validation = read('featuresall_train.txt')
    train_x = train[:, 1:]
    train_y = train[:, 0]
    train_y = train_y.reshape(train_y.shape[0], 1)
    validation_x = validation[:, 1:]
    validation_y = validation[:, 0]
    validation_y = validation_y.reshape(validation_y.shape[0], 1)

    train_x = np.abs(train_x)
    validation_x = np.abs(validation_x)

    # normalize
    base = 10.
    
    train_x         = np.log(train_x + 10.) / np.log(base)
    validation_x    = np.log(validation_x + 10.) / np.log(base)
    train_x         = tf.keras.utils.normalize(train_x)
    validation_x    = tf.keras.utils.normalize(validation_x)
    
    # build up model
    model = get_model()

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # train

    print(train_x.shape)
    print(train_y.shape)
    print(validation_x.shape)
    print(validation_y.shape)

    history = model.fit(train_x, train_y, epochs = 1000, validation_data = (validation_x, validation_y), shuffle = True)
    #with open('model_history.txt','w') as file:
    #    file.write(str(history.history))

    model.save('model_allv0', include_optimizer = True)

    return 0


main()