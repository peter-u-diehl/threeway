import scipy.ndimage as sp
import numpy as np
import matplotlib.cm as cm
import matplotlib
import pylab
import os


def randomDelay(minDelay, maxDelay):
    return np.random.rand()*(maxDelay-minDelay) + minDelay
        
        
def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos

        
def sparsenMatrix(weightBaseMatrix, delayBaseMatrix, pConn):
    weightMatrix = np.zeros(weightBaseMatrix.shape)
    delayMatrix = np.zeros(delayBaseMatrix.shape)
    numWeights = 0
    numTargetWeights = delayBaseMatrix.shape[0] * delayBaseMatrix.shape[1] * pConn
    weightList = [0]*int(numTargetWeights)
    delayList = [0]*int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*delayBaseMatrix.shape[0]), np.int32(np.random.rand()*delayBaseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = weightBaseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], weightBaseMatrix[idx])
            delayMatrix[idx] = delayBaseMatrix[idx]
            delayList[numWeights] = (idx[0], idx[1], delayBaseMatrix[idx])
            numWeights += 1
    weightList = sorted(sorted(weightList, key=lambda weight: weight[1]), key=lambda weight: weight[0])
    delayList = sorted(sorted(delayList, key=lambda weight: weight[1]), key=lambda weight: weight[0])
    return weightMatrix, weightList, delayMatrix, delayList
        
        
def createRandomWeights(dataPath):
    np.random.seed(70)
    nE = 1600
    nI = nE/4
    weight = {}
    weight['ee_input'] = 0.5 # 0.10
    weight['ei_input'] = 0.2 # 0.08
    weight['ee'] = 0.2
    weight['ei'] = 0.2
    weight['ie'] = 4.0
    weight['ii'] = 0.4
    delay = {}
    delay['ee_input'] = (4,10)#(0,10)
    delay['ei_input'] = (0,4)
    delay['ee'] = (2,5)#(0,5)
    delay['ei'] = (0,1)#(0,2)
    delay['ie'] = (0,1)
    delay['ii'] = (0,2)
    pConn = {}
    pConn = 0.1
    
            
    print 'create random connection matrices and delays'
    inputConns = ['XeAe', 'YeBe', 'ZeCe', ]
    connNameList = ['XeAe', 'YeBe', 'ZeCe', 
                    'AeHe', 'BeHe', 'CeHe', 
                    'HeAe', 'HeBe', 'HeCe',
                    'AeAe', 'BeBe', 'CeCe', 'HeHe',]
    connNameList += ['XeAi', 'YeBi', 'ZeCi', 
                    'AeHi', 'BeHi', 'CeHi', 
                    'HeAi', 'HeBi', 'HeCi',
                    'AeAi', 'BeBi', 'CeCi', 'HeHi',]
    connNameList += ['AiAe', 'BiBe', 'CiCe', 'HiHe']
    connNameList += ['AiAi', 'BiBi', 'CiCi', 'HiHi']
    
    for name in connNameList:
        if name[1]=='e':
            nSrc = nE
        else:
            nSrc = nI
        if name[3]=='e':
            nTgt = nE
        else:
            nTgt = nI
            
        weightMatrix = np.random.random((nSrc, nTgt))
        delayMatrix = np.random.random((nSrc, nTgt))
        
        if name in inputConns:
            weightMatrix *= weight['ee_input']
            delayMatrix *= (delay['ee_input'][1] - delay['ee_input'][0])
            delayMatrix += delay['ee_input'][0]
        else:
            weightMatrix *= weight[name[1]+name[3]]
            delayMatrix *= (delay[name[1]+name[3]][1] - delay[name[1]+name[3]][0])
            delayMatrix += delay[name[1]+name[3]][0]
        
        if name[2]=='H' and not name[0]=='H':
            weightMatrix *= 0.66
            
        weightMatrix, weightList, delayMatrix, delayList = sparsenMatrix(weightMatrix, delayMatrix, pConn)
        np.save(dataPath + 'random/' + name, weightList)
        np.savetxt(dataPath + 'random/' + name + '.txt', weightList)
        np.save(dataPath + 'random_delay/' + name, delayList)
        np.savetxt(dataPath + 'random_delay/' + name + '.txt', delayList)
    
    
    
    
if __name__ == "__main__":
    createRandomWeights(os.getcwd() + '/')
    
    
    









