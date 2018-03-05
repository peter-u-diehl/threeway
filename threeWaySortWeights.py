'''
Created on Sep 29, 2014

@author: peter
'''

import numpy as np
import os


def getWeightsFromFile(fileName,nE):
    readout = np.load(fileName)
    value_arr = np.zeros((nE, nE))
    value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    
#     print value_arr
#     print fileName, nSrc, nTgt
#     figure()
#     im2 = imshow(value_arr, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#     cbar2 = colorbar(im2)
#     title(fileName)
#     show()
    return value_arr, readout


def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos


def sortInput(wEE,nE):
    wEE[np.isnan(wEE)] = 0
    pop_val = [computePopVector(wEE[:nE,i]) for i in xrange(nE)]
    reverse_order = list(np.argsort(pop_val, axis=0))
#    print 'reverse_order:', reverse_order
    order = np.asarray([reverse_order.index(i) for i in xrange(len(reverse_order))])
    return order


def saveSorted(nE, nI, srcWeightPath, tgtWeightPath, 
               ending, inputConnectionNames, interPopConnectionNames):
    order = {}
    
    for inputConnName in inputConnectionNames:
        connName = inputConnName[0] + 'e' + inputConnName[1] + 'e'    #sort only connections between excitatory neurons
        print 'sort input E->E connection', connName
        weightMatrix, weighList = getWeightsFromFile(srcWeightPath + connName + ending + '.npy',nE)
        order[connName[2]] = sortInput(weightMatrix,nE)
        connListSparse = [(entry[0], order[connName[2]][entry[1]], entry[2]) for entry in weighList]
        np.save(tgtWeightPath + connName + ending, connListSparse)
        
        connName = inputConnName[0] + 'e' + inputConnName[1] + 'i'    #sort only connections between excitatory neurons
        print 'sort input E->I connection', connName
        weightMatrix, weighList = getWeightsFromFile(srcWeightPath + connName + ending + '.npy',nE)
        np.save(tgtWeightPath + connName + ending, weighList)
        
        
    order['H'] = np.asarray(range(nE))

    for popName in ['A', 'B', 'C']:
        connTypes = ['ee', 'ei', 'ie', 'ii']
        for connType in connTypes:
            print 'sort recurrent connection', connType, ' of population ', popName
            connName = popName + connType[0] + popName + connType[1]    #sort only connections between excitatory neurons
            weightMatrix, weighList = getWeightsFromFile(srcWeightPath + connName + ending + '.npy',nE)
            if connType == 'ee':
                connListSparse = [(order[popName][entry[0]], order[popName][entry[1]], entry[2]) for entry in weighList]
            if connType == 'ei':
                connListSparse = [(order[popName][entry[0]], entry[1], entry[2]) for entry in weighList]
            if connType == 'ie':
                connListSparse = [(entry[0], order[popName][entry[1]], entry[2]) for entry in weighList]
            if connType == 'ii':
                connListSparse = [(entry[0], entry[1], entry[2]) for entry in weighList]
            np.save(tgtWeightPath + connName + ending, connListSparse)
             
        for interConnName in interPopConnectionNames:
            if popName == interConnName[0]:
                connName = interConnName[0] + 'e' + interConnName[1] + 'e'    #sort only connections between excitatory neurons
                print 'sort inter area E->E connection', connName
                weightMatrix, weightList = getWeightsFromFile(srcWeightPath + connName + ending + '.npy',nE)
                connListSparse = [(order[popName][entry[0]], order[interConnName[1]][entry[1]], entry[2]) for entry in weightList]
                np.save(tgtWeightPath + connName + ending, connListSparse)
                
                connName = interConnName[0] + 'e' + interConnName[1] + 'i'    #sort only connections between excitatory neurons
                print 'sort inter area E->I connection', connName
                weightMatrix, weightList = getWeightsFromFile(srcWeightPath + connName + ending + '.npy',nE)
                connListSparse = [(order[popName][entry[0]], entry[1], entry[2]) for entry in weightList]
                np.save(tgtWeightPath + connName + ending, connListSparse)
                
            if popName == interConnName[1]:
                connName = interConnName[0] + 'e' + interConnName[1] + 'e'    #sort only connections between excitatory neurons
                print 'sort inter area E->E connection', connName
                weightMatrix, weightList = getWeightsFromFile(srcWeightPath + connName + ending + '.npy',nE)
                connListSparse = [(entry[0], order[interConnName[1]][entry[1]], entry[2]) for entry in weightList]
                np.save(tgtWeightPath + connName + ending, connListSparse)
                
                connName = interConnName[0] + 'e' + interConnName[1] + 'i'    #sort only connections between excitatory neurons
                print 'sort input E->I connection', connName
                weightMatrix, weighList = getWeightsFromFile(srcWeightPath + connName + ending + '.npy',nE)
                np.save(tgtWeightPath + connName + ending, weighList)
            

    print 'sort H using AeHe and BeHe'
    weightMatrixAH, unused_weighListAH = getWeightsFromFile(tgtWeightPath + 'AeHe' + ending + '.npy',nE)
    weightMatrixBH, unused_weighListBH = getWeightsFromFile(tgtWeightPath + 'BeHe' + ending + '.npy',nE)
    popVecsH = [(computePopVector(weightMatrixAH[:, i]) + computePopVector(weightMatrixBH[:, i])) % 1. for i in xrange(nE)]
    inverse_orderH = list(np.argsort(popVecsH, axis=0))
    order['H'] = np.asarray([inverse_orderH.index(i) for i in xrange(len(inverse_orderH))])
#     print popVecsH 
#     print order['H']
    
    popName = 'H'
    connTypes = ['ee', 'ei', 'ie', 'ii']
    for connType in connTypes:
        print 'sort recurrent connection', connType, ' of population ', popName
        connName = popName + connType[0] + popName + connType[1]    #sort only connections between excitatory neurons
        weightMatrix, weighList = getWeightsFromFile(srcWeightPath + connName + ending + '.npy',nE)
        if connType == 'ee':
            connListSparse = [(order[popName][entry[0]], order[popName][entry[1]], entry[2]) for entry in weighList]
        if connType == 'ei':
            connListSparse = [(order[popName][entry[0]], entry[1], entry[2]) for entry in weighList]
        if connType == 'ie':
            connListSparse = [(entry[0], order[popName][entry[1]], entry[2]) for entry in weighList]
        if connType == 'ii':
            connListSparse = [(entry[0], entry[1], entry[2]) for entry in weighList]
        np.save(tgtWeightPath + connName + ending, connListSparse)
         
    for interConnName in interPopConnectionNames:
        if popName == interConnName[0]:
            connName = interConnName[0] + 'e' + interConnName[1] + 'e'    #sort only connections between excitatory neurons
            print 'sort inter area E->E connection', connName
            weightMatrix, weightList = getWeightsFromFile(tgtWeightPath + connName + ending + '.npy',nE)
            connListSparse = [(order[popName][entry[0]], entry[1], entry[2]) for entry in weightList]
            np.save(tgtWeightPath + connName, connListSparse)
            
            connName = interConnName[0] + 'e' + interConnName[1] + 'i'    #sort only connections between excitatory neurons
            print 'sort inter area E->I connection', connName
            weightMatrix, weightList = getWeightsFromFile(tgtWeightPath + connName + ending + '.npy',nE)
            connListSparse = [(order[popName][entry[0]], entry[1], entry[2]) for entry in weightList]
            np.save(tgtWeightPath + connName, connListSparse)
            
        if popName == interConnName[1]:
            connName = interConnName[0] + 'e' + interConnName[1] + 'e'    #sort only connections between excitatory neurons
            print 'sort inter area E->E connection', connName
            weightMatrix, weightList = getWeightsFromFile(tgtWeightPath + connName + ending + '.npy',nE)
            connListSparse = [(entry[0], order[popName][entry[1]], entry[2]) for entry in weightList]
            np.save(tgtWeightPath + connName, connListSparse)


if __name__ == "__main__":
    nE = 1600
    nI = nE/4
    
    dataPath = os.getcwd() + '/'
    srcWeightPath = dataPath + 'weights/'
    tgtWeightPath = dataPath + 'sortedWeights/'
#     srcWeightPath = dataPath + 'random/'
#     tgtWeightPath = dataPath + 'randomSortedWeights/'
    ending = ''
    inputConnectionNames = ['XA', 'YB', 'ZC']
    interPopConnectionNames = ['AH', 'BH', 'CH',
                              'HA', 'HB', 'HC',
                              ]
    
    saveSorted(nE, nI, srcWeightPath, tgtWeightPath, 
               ending, inputConnectionNames, interPopConnectionNames)












