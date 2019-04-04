import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

# Create a model with a couple of layers
def fully_connected_model():
    model = Sequential()
    if (isCNN):
        model.add(Conv2D(1, kernel_size=(10, 10), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(Flatten())
    model.add(Dense(numPixels, input_dim=numPixels, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(numClasses, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model

# Arguments: X is training feature vectors
#            T is training class labels (not needed other than for scatter plot)
# Returns:   Mean vector of X
#            Eigen vectors, as rows, with max on top
#            Principal components of X
def performPCA(X, T, showScatterPlot=False, showSeparatePlots=False):
    # PERFORM the XZCVP process
    # X -> Z
    muX = np.mean(X, axis=0) # get means of columns, by setting axis=0
    Z = X - muX
    
    # Z -> C
    C = np.cov(Z.T) # alter input such that each row is a feature
    
    # C -> V
    [eigVal, V] = LA.eigh(C)
    
    eigVal = np.flipud(eigVal) # get largest value to beginning
    V = np.flipud(V.T)         # get eigen vectors to be rows, with largest one first
    row = V[0,:] #Check once again
    np.dot(C,row) - (eigVal[0]*row) #If the matrix product C.row is the same as λ[0]*row, this should evaluate to [0,0,0]
    
    # V -> P
    P = np.dot(Z, V.T) # Nxd DOT dxd = Nxd

    # Draw SCATTER PLOT
    if (showScatterPlot):
        reducedP = P[:, 0:2]
        fig = plt.figure()
        ax = fig.add_subplot(111, facecolor='black')
        
        for digit in np.unique(T):
            ix = np.where(T == digit)
            ax.scatter(reducedP[ix,0], reducedP[ix,1], s=5, c=mapOfColors[digit], label='Digit '+str(digit), linewidths=0, marker="o")
        ax.set_aspect('equal')
        plt.legend(markerscale=5.0)
        plt.show()

    # Draw SCATTER PLOT (separate for each class)
    if (showSeparatePlots):
        plt.close('all')
        fig = plt.figure()
        
        reducedP = P[:, 0:2]
        
        for digit in np.unique(T):
            ix = np.where(T == digit)
            ax = fig.add_subplot(plotLoc[digit], facecolor='black')
            ax.scatter(reducedP[ix,0], reducedP[ix,1], s=5, c=mapOfColors[digit], label='Digit '+str(digit), linewidths=0, marker="o")
            ax.set_title(mapOfEmotion[digit])
            ax.set_xbound(-6000, 6000)
            ax.set_ybound(-3500, 3500)
        ax.set_aspect('equal')
        #plt.legend(markerscale=5.0)
        plt.show()

    return muX, V, P
    

### Classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
mapOfEmotion = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
mapOfColors = {0: [1,0,0,0.5], 1: [0,1,0,0.5], 2: [0,0,1,0.5], 3:[1,1,0,0.5], 4:[0,1,1,0.5], 5:[1,0,1,0.5], 6:[0.5,0.5,0.5,0.5]}
plotLoc = {0: 241, 1: 242, 2: 243, 3:244, 4:245, 5:246, 6:247}

trainOnPCs = False
isCNN = True

## Data Loading and Splitting
dataFolderPath = "D:\\python_ML\\Project\\"
data=pd.read_csv(dataFolderPath + "FaceRecognition.csv")

data=data.drop(columns='Unnamed: 0')

Train_data=data[data['X2306']=='Training'].drop(columns='X2306')
Test_data=data[data['X2306']=='PrivateTest'].drop(columns='X2306')
Train_labels=Train_data.iloc[:,0].values
Test_labels=Test_data.iloc[:,0].values
Train_data=Train_data.drop(columns='X1').values
Test_data=Test_data.drop(columns='X1').values

Val_data=data[data['X2306']=='PublicTest'].drop(columns='X2306')
Val_labels=Val_data.iloc[:,0].values
Val_data=Val_data.drop(columns='X1').values

# specify type for both data and labels
Train_data = Train_data.astype(dtype=np.float32)
Val_data = Val_data.astype(dtype=np.float32)
Test_data = Test_data.astype(dtype=np.float32)

Train_labels = Train_labels.astype(dtype=np.uint8)
Val_labels = Val_labels.astype(dtype=np.uint8)
Test_labels = Test_labels.astype(dtype=np.uint8)

imgX, imgY = 48, 48

# Preprocess the training data
if (trainOnPCs):
    nPCDim = 300
    # Peform PCA on the training data
    muTrain, V, PTrain = performPCA(Train_data, Train_labels, False, False)
    reducedV = V[0:nPCDim, :]
    
    # convert training data
    Train_x = PTrain[:, 0:nPCDim]
    
    # convert val data
    ZVal = Val_data - muTrain
    Val_x = np.dot(ZVal, reducedV.T)
    
    # convert test data
    ZTest = Test_data - muTrain
    Test_x = np.dot(ZTest, reducedV.T)
    
    # now, normalize all 3 sets
    # low = min(np.amin(Val_x), np.amin(Train_x), np.amin(Test_x))
    # high = max(np.amax(Val_x), np.amax(Train_x), np.amax(Test_x))
    # 
    # Train_x = (Train_x - low) / (high-low)
    # Val_x = (Val_x - low) / (high-low)
    # Test_x = (Test_x - low) / (high-low)
else:
    
    if (isCNN):
        Train_data = Train_data.reshape(Train_data.shape[0], imgX, imgY, 1)
        Val_data = Val_data.reshape(Val_data.shape[0], imgX, imgY, 1)
        Test_data = Test_data.reshape(Test_data.shape[0], imgX, imgY, 1)
        
        input_shape = (imgX, imgY, 1)
    
    Train_x = Train_data/255
    Val_x = Val_data/255
    Test_x = Test_data/255

# Keslerize the target labels
Train_y = np_utils.to_categorical(Train_labels)
Val_y = np_utils.to_categorical(Val_labels)
Test_y = np_utils.to_categorical(Test_labels)

numPixels = imgX * imgY
numClasses = Train_y.shape[1]
# instantiate a model
mdl = fully_connected_model()

# fit the model to the training data (provide a validation set as well)
np.random.seed(0)
mdl.fit(Train_x, Train_y, validation_data=(Val_x, Val_y), epochs=100, batch_size=32, verbose=2)

scores = mdl.evaluate(Test_x, Test_y, verbose=0)
print('Accuracy on the PRIVATE TEST set is ', scores[1])
# on original comp
#   Accuracy of 0.376706603519031 on the PRIVATE TEST set.
# try other batch sizes?

# try training on PCs?
#   360 PCs, norm. -> 24% and stays unchanged with iterations!
#   360 PCs, no norm. -> 33.49%
# 

# testResCM = confusion_matrix(Test_labels, predLabelTest)
# print('Confusion matrix is:\n', testResCM)
# acc = testResCM.diagonal().sum() / testResCM.sum()
# print('Accuracy over PRIVATE TEST SET is ', acc)
# print('PPV values for the classes are:', resCm.diagonal() / resCm.sum(axis=0))
# print('Sensitivity values for the classes are:', resCm.diagonal() / resCm.sum(axis=1))