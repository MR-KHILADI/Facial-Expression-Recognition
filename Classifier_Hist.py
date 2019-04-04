import numpy as np
import matplotlib.pyplot as plt
from pandas import ExcelWriter, ExcelFile, read_excel, DataFrame
from openpyxl import load_workbook

def binForValue(xValue, B, xmin, xmax):
    computedBin = np.round((B-1)*(xValue-xmin)/(xmax-xmin)).astype('int32')
    return np.clip(computedBin, 0, (B-1))
    
def buildHistogramClassifier(X, T, B, verbose=False):
    
    hist = {} # an empty dictionary, key is set of indices, value is count for each class at that position
    
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    nDistinctClasses = len(np.unique(T))
    minVals = np.amin(X, axis=0) # find mins for each column
    maxVals = np.amax(X, axis=0)
    
    binIndices = np.zeros((nSamples, nFeatures), dtype=np.uint8)
    for iFeature in np.arange(nFeatures):
        binIndices[:,iFeature] = binForValue(X[:,iFeature], B, minVals[iFeature], maxVals[iFeature])

    for i, binIndexSet in enumerate(binIndices):
        # binIndexSet is the key into the dict
        # if it already exists, get the value
        # else create a new value (vector of length nClass)
        # based on G[i], increment corresponding position in vector
        tupleOfIndices = tuple(binIndexSet)
        classIndex = T[i] # since labels are 0-based numbers
        if tupleOfIndices not in hist.keys():
            hist[tupleOfIndices] = np.zeros((nDistinctClasses), dtype=np.uint16)
        hist[tupleOfIndices][classIndex] += 1
        
        if (verbose):
            if (hist[tupleOfIndices][classIndex] > 8000):
                print('Nearing the limit for ', classIndex)
            if (np.remainder(i, 1000) == 0):
                print('Done with sample# ', i)

    return hist, minVals, maxVals


def predictWithHistogram(H, B, minVals, maxVals, nDistinctClasses, queries):
    # based on QUERIES, grab numbers from histograms and produce prob
    predLabel = []
    predProb = np.zeros(len(queries))
    
    nQueries = queries.shape[0]
    nFeatures = queries.shape[1]
        
    binIndices = np.zeros((nQueries, nFeatures), dtype=np.uint8)
    for iFeature in np.arange(nFeatures):
        binIndices[:,iFeature] = binForValue(queries[:,iFeature], B, minVals[iFeature], maxVals[iFeature])

    for i, binIndexSet in enumerate(binIndices):
        # binIndexSet is the key into the dict
        # if it already exists, get the value
        # else create a new value (vector of length nClass)
        # based on G[i], increment corresponding position in vector
        tupleOfIndices = tuple(binIndexSet)
        
        if tupleOfIndices in H.keys():
            countForClasses = H[tupleOfIndices]
            likelyClass = np.argmax(countForClasses)
        else:
            likelyClass = -1
        predLabel.append(likelyClass)
        
    return predLabel, predProb