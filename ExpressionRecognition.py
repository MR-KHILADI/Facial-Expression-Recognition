import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy.linalg as LA

#from Classifier_Hist_Bayes_2D import (buildBayesianClassifier, predictWithBayesian)

def vectortoimg(v,show=True):
    plt.imshow(v.reshape(48, 48),interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()

# Arguments: X is training feature vectors
#            T is training class labels (not needed other than for scatter plot)
# Returns:   Mean vector of X
#            Eigen vectors, as rows, with max on top
#            Principal components of X
def performPCA(X, T):
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
    # reducedP = P[:, 0:2]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, facecolor='black')
    #
    # for digit in np.unique(T):
    #     ix = np.where(T == digit)
    #     ax.scatter(reducedP[ix,0], reducedP[ix,1], s=5, c=mapOfColors[digit], label='Digit '+str(digit), linewidths=0, marker="o")
    # ax.set_aspect('equal')
    # plt.legend(markerscale=5.0)
    # plt.show()

    return muX, V, P

def pdfLog(x, mu, cvMtx):
    
    diffFromMean = (x - mu)
    cvMtxInv = np.linalg.inv(cvMtx)
    powOfE = diffFromMean.dot(cvMtxInv)
    powOfE = powOfE.dot(diffFromMean.T)
    numerPowOfE = -.5 * powOfE
    
    s, logDet = np.linalg.slogdet(cvMtx)
    if s == 1:
        denomPowOfE = np.log(2*np.pi) + logDet*0.5
    else:
        print('Determinant of cvMtx is 0 or negative!')
    
    return (numerPowOfE-denomPowOfE)

def buildBayesianClassifier(X, T):
    
    # split training data into class-specific sets
    XByClass = []
    nByClass = []
    muByClass = []
    cvMtxByClass = []
    
    for cLabel in np.unique(T):
        ix = np.where(T == cLabel)
        ix = ix[0] # as ix is initially a tuple with one element
        XForAClass = X[ix,:]
        XByClass.append(XForAClass)
    
        # get SAMPLE SIZE
        nByClass.append(len(XForAClass))

        # get MEAN VECTOR
        mu = np.mean(XForAClass, axis=0)
        mu = mu.reshape(1, mu.shape[0])
        muByClass.append(mu)
    
        # get CovarianceMatrix
        cvMtx = np.cov(XForAClass.T) # transpose so that each column is a single obsv.
        cvMtxByClass.append(cvMtx)

    return XByClass, nByClass, muByClass, cvMtxByClass

def predictWithBayesian(nByClass, muByClass, cvMtxByClass, classLabels, queries):
    # for each QUERY, compute bayesian numbers and produce prob
    predLabel = []
    predProb = np.zeros(len(queries))
    for iQuery, xQueryVal in enumerate(queries):
        
        #estByClass = [0.]*len(classLabels)
        estLogByClass = [0.]*len(classLabels)
        for iClass, labelVal in enumerate(classLabels):
           estLogByClass[iClass] = pdfLog(xQueryVal, muByClass[iClass], cvMtxByClass[iClass]) + np.log(nByClass[iClass])
           #estByClass[iClass] = np.exp(estLogByClass[iClass])
        
        #probByClass = estByClass / np.sum(estByClass)
        #likelyClass = np.argmax(probByClass)
        likelyClass = np.argmax(estLogByClass)
        
        predLabel.append(classLabels[likelyClass])
        #predProb[iQuery] = probByClass[likelyClass]
        
        if (np.remainder(iQuery,50) == 0):
            print('Done with query# ', iQuery) 
            
    return predLabel, predProb


### Classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
mapOfEmotion = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
mapOfColors = {0: [1,0,0,0.5], 1: [0,1,0,0.5], 2: [0,0,1,0.5], 3:[1,1,0,0.5], 4:[0,1,1,0.5], 5:[1,0,1,0.5], 6:[0.5,0.5,0.5,0.5]}

## Data Loading and Splitting
#data=pd.read_csv(r"F:\ML_Intro_Python\Project\FaceRecognition.csv")
data=pd.read_csv(r"D:\python_ML\Project\FaceRecognition.csv")

data=data.drop(columns='Unnamed: 0')

Train_data=data[data['X2306']=='Training'].drop(columns='X2306')
Test_data=data[data['X2306']=='PrivateTest'].drop(columns='X2306')
Train_labels=Train_data.iloc[:,0].values
Test_labels=Test_data.iloc[:,0].values
Train_data=Train_data.drop(columns='X1').values
Test_data=Test_data.drop(columns='X1').values

# for x in np.arange(5):
#     nImgToDraw = random.randint(0, X.shape[0]-1)
#     print(mapOfEmotion[Train_labels[nImgToDraw]])
#     vectortoimg(Train_data[nImgToDraw])

# downsample from 64bit int to an 8bit uint
Train_data = Train_data.astype(dtype=np.uint8)
Train_labels = Train_labels.astype(dtype=np.uint8)

muTrain, V, PTrain = performPCA(Train_data, Train_labels)

# 2 gives us accuracy of 25.57%
# 19 gives us accuracy of 32.79%
# 50 gives us accuracy of 38.31%
# 100 (after switching to log scale in pdf calc) gives 39.87%
# 150 (only predicting label) gives 41.60%
# 200 gives 42.24%
# 250 gives 43.02%
# 300 gives 43.32%
# 350 gives 43.80%
# 375 gives 44.16%
# 400 gives 23.54%
# 500 gives 23.01%

nPCDim = 390
reducedV = V[0:nPCDim, :]

# try recovering some of the images, to see how they look
reducedP = PTrain[:, 0:nPCDim]
xRec = np.dot(reducedP, reducedV) + muTrain

# randomly pick and view few images and their label
print("Checking multiple training images by plotting them.\nBe patient:")
plt.close('all')
fig = plt.figure()
nrows=8
ncols=4
for row in range(nrows):
    for col in range(ncols):
        ax = plt.subplot(nrows, ncols*2, row*ncols*2 + 2*col + 1)
        
        nImgToDraw = random.randint(0, Train_data.shape[0]-1)
        vectortoimg(Train_data[nImgToDraw], show=False)
        ax.set_title(mapOfEmotion[Train_labels[nImgToDraw]])
        
        ax = plt.subplot(nrows, ncols*2, row*ncols*2 + 2*col + 2)
        vectortoimg(xRec[nImgToDraw], show=False)
plt.show()

# now, get a Bayesian classifier for the training data and labels
TrainByClass, nByClass, muByClass, cvMtxByClass = buildBayesianClassifier(reducedP, Train_labels)

# convert test set into queries (in the form of PCs)
Test_data = Test_data.astype(dtype=np.uint8)
ZTest = Test_data - muTrain
PTest = np.dot(ZTest, reducedV.T)

# now, predict on the test set with Bayesian classifier
predLabelTest, predProbTest = predictWithBayesian(nByClass, muByClass, cvMtxByClass, np.unique(Train_labels), PTest)

res = (predLabelTest==Test_labels)
print('Accuracy is ', res.sum()/len(Test_labels))