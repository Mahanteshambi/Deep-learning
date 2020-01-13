# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 17:09:02 2018

@author: Ambi
"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random, time


kMeansOPDir = 'output/kmeans/'

def getFeatureVectors(dir):
    featureFiles = [join(dir, file) for file in listdir(dir) if isfile(join(dir, file))]
    featureFiles.sort()
    fileCount = 0
    titleRow = ["Index", "Label", "Data", "Feature Vector"]
    labelList = list()
    characterDataList = list()
    featureVectorList = list()
    for file in featureFiles:
        '''if fileCount == 1:
            break'''
        fileCount += 1
        print("Parsing %s" % (file))
        data = pd.read_csv(file)
        dataFrame = pd.DataFrame(data, columns = titleRow)
        for i in range(len(dataFrame)):
            label = dataFrame['Label'][i].replace('[', '').replace(']', '').split(',')
            characterData =  dataFrame['Data'][i].replace('[', '').replace(']', '').split(',')
            featureVector =  dataFrame['Feature Vector'][i].replace('[', '').replace(']', '').split(',')
            
            labelList.append(label)
            characterDataList.append(characterData)
            featureVectorList.append(featureVector)
            
    for i in range(len(characterDataList)):
        characterDataList[i] = [int(float(value)) for value in characterDataList[i]]
        characterDataList[i] = np.array(characterDataList[i]).reshape(28, 28)
        featureVectorList[i] = [float(value) for value in featureVectorList[i]]
        labelList[i] = [float(value) for value in labelList[i]]
    return labelList, characterDataList, featureVectorList

    

    
def evaluateKMeans(clusterData):
    print('')
    totalAccuracy = 0
    for j in range(10):
        labelsClusteredList = []
        for i in range(10):
            labelsClusteredList.append(0)
        for i in range(len(clusterData[j])):
            labelsClusteredList[clusterData[j][i][1].index(1.0)] += 1
        clusterAccuracy = max(labelsClusteredList) / sum(labelsClusteredList) * 100.0
        totalAccuracy += clusterAccuracy
    print("KMeans clustering Accuracy " + str(totalAccuracy / 10))



class KMeansClustering:
    
    def __init__(self, featureVectorList, labels):
        self.featureVectorList = featureVectorList
        self.labels = labels
        
    def fitAndPredict(self):
        startTime = time.time()
        k = 10
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(featureVectorList)
        self.kMeanslabels = kmeans.predict(featureVectorList)
        print("Clustered using KMeans in [%.3f seconds]" % (time.time() - startTime))
        self.kMeanscentroids = kmeans.cluster_centers_
      
    def findOptimalK(self):
        wcss = list()
        for k in range(1, 15):
            kmeans = KMeans(n_clusters=k)
            kmeans = kmeans.fit(featureVectorList)
            wcss.append(kmeans.inertia_)
            
        plt.figure(figsize=(15, 6))
        plt.plot(range(1, 15), wcss, marker = "o")
        
    def dumpResults(self):
        clusterData = []
        for i in range(10):
            clusterData.append([])
        for i in range(len(self.kMeanslabels)):
            clusterData[self.kMeanslabels[i]].append([characterDataList[i], labelList[i]])
            
        for l in range(0):
            print("Plotting for label %d" % (l))
            fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True)
            fig.set_figheight(15)
            fig.set_figwidth(15)
            count = 0
            randomNoList = random.sample(range(0, len(clusterData[l])), 100)
            for i in range(10):
                for j in range(10):
                    axes[i][j].imshow(clusterData[l][randomNoList[count]][0])
                    count += 1
            fig.savefig(kMeansOPDir + 'kmeans_cluster' + str(l) + '.png')
        return clusterData
    
class AgglomerativeClustering():
    
    def __init__(self, featureVectorList, labels):
        self.featureVectorList = featureVectorList
        self.labels = labels
        
    def fitAndPredict(self):
        startTime = time.time()
        k = 10
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(featureVectorList)
        self.kMeanslabels = kmeans.predict(featureVectorList)
        print("Clustered using KMeans in [%.3f seconds]" % (time.time() - startTime))
        self.kMeanscentroids = kmeans.cluster_centers_

#if __name__ == "__main__":
startTime = time.time()
labelList, characterDataList, featureVectorList = getFeatureVectors('output/colab_features/')
print("Extracted details in [%.3f seconds]" % (time.time() - startTime))

kMeansObj = KMeansClustering(featureVectorList, labelList)
kMeansObj.findOptimalK()
kMeansObj.fitAndPredict()


startTime = time.time()
clusterData = kMeansObj.dumpResults()
print("Dumped results in [%.3f seconds]" % (time.time() - startTime))
evaluateKMeans(clusterData)
    

    


