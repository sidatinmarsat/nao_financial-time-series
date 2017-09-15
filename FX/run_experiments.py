import sys
import numpy as np

#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import sklearn.linear_model
import sklearn.metrics

from scipy.stats import spearmanr as cor

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Merge
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from sklearn import preprocessing

def makeDNN(in_size, dropout=0.2):
    l2_lam = 5e-07 
    l1_lam = 1e-08 
    model = Sequential()
    model.add(Flatten(input_shape=(in_size)))
    model.add(Dense(150, kernel_regularizer=regularizers.l1(l1_lam)))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))
    model.add(Dense(100, kernel_regularizer=regularizers.l1(l1_lam))) 
    model.add(Activation("relu"))
    model.add(Dropout(dropout))
    model.add(Dense(50, kernel_regularizer=regularizers.l1(l1_lam))) 
    model.add(Activation("relu"))
    model.add(Dense(3, kernel_regularizer=regularizers.l1(l1_lam)))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def makeCNN(in_size, dropout=0.2):
    nkernels = [100,50]
    l2_lam = 5e-07 
    l1_lam = 1e-08 
    model = Sequential()
    model.add(Conv2D(nkernels[0], kernel_size=(1,8), strides=(1,1), padding='same', input_shape=in_size, kernel_regularizer=regularizers.l2(l2_lam)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,4), strides=(1,4)))
    model.add(Dropout(0.2))

    model.add(Conv2D(nkernels[1], kernel_size=(1,8), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(l2_lam)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,4), strides=(1,4)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=regularizers.l1(l1_lam)))
    model.add(Activation('relu'))
    model.add(Dense(3, kernel_regularizer=regularizers.l1(l1_lam)))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    print "python run_experiemnts.py choice suffix epochs train/test"
    if sys.argv[1] == "train":
        suffix = sys.argv[2] #"-2017_01_01-2017_07_03.npy"
        X1 = np.load("X"+suffix)
        X2 = np.load("X2"+suffix)
        X3 = np.load("X3"+suffix)
        Y = np.load("Y"+suffix)
        
        choice = sys.argv[3]
        ep = int(sys.argv[4])
        filepath = "best-"+choice+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        X = np.concatenate([X1,X2,X3], axis=2)
        index = range(X.shape[0])
        np.random.shuffle(index)
        X = X[index]
        Y = Y[index]
        
        print "Feature size:", X.shape[1:]
        
        if choice == "DNN":
            model1 = makeDNN(X.shape[1:])
            model1.fit(X, Y,
                              epochs=ep, 
                              batch_size=64,
                              callbacks=callbacks_list,
                              validation_split=0.2)
        elif choice == "CNN":
            X = np.expand_dims(X, axis=1)
            model2 = makeCNN(X.shape[1:])
            model2.fit(X, Y,
                              epochs=ep, 
                              batch_size=64,
                              callbacks=callbacks_list,
                              validation_split=0.2)
        else:
            print "Invalid choice"
            return -1
        
        
    elif sys.argv[1] == "test":
        suffix = sys.argv[2] #"-2017_01_01-2017_07_03.npy"
        X1 = np.load("X"+suffix)
        X2 = np.load("X2"+suffix)
        X3 = np.load("X3"+suffix)
        Y = np.load("Y"+suffix)
        X = np.concatenate([X1,X2,X3], axis=2)
                 
        choice = sys.argv[3]
        print "testing", choice
        
        model = load_model("best-"+choice+".hdf5")
                 
        if choice == "CNN":
            X = np.expand_dims(X, axis=1)
        
        temp = model.predict(X)
        np.save("prediction-"+choice, temp)
                 
        randtemp = np.zeros(temp.shape)
        for i in range(randtemp.shape[0]):
            randtemp[i,np.random.randint(3)] = 1
        
        labels = ["up", "down", "negative"]
        for i in range(len(labels)):
            prediction = (np.argmax(temp, axis=1)==i)
            #print prediction
            print labels[i]+":"
            print "accuracy:", sklearn.metrics.accuracy_score(prediction, Y[:, i]),\
            "Random:", sklearn.metrics.accuracy_score(prediction, randtemp[:, i]) 
            print "precision:", sklearn.metrics.precision_score(prediction, Y[:, i]),\
            "Random:", sklearn.metrics.precision_score(prediction, randtemp[:, i]) 
            print "recall:", sklearn.metrics.recall_score(prediction, Y[:, i]),\
            "Random:", sklearn.metrics.recall_score(prediction, randtemp[:, i]) 

if __name__ == "__main__":
    main()
