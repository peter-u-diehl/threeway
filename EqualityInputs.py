import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

def setNewInput(net,j):
    '''
        Assumes that the input net contains three input groups named X, Y, and Z which are stored in a dictionary inputGroups.
    '''
    for i,name in enumerate(net.inputPopulationNames):
        if name == 'X':
            net.popValues[j,i] = np.random.rand();
            rates = net.createTopoInput(net.nE, net.popValues[j,i])
        else:
            if net.testMode:
                rates = np.ones(net.nE)  * 0
            elif name == 'Y':
                net.popValues[j,i] = (net.popValues[j,0])
                rates = net.createTopoInput(net.nE, net.popValues[j,i])
            elif name == 'Z':
                net.popValues[j,i] = (net.popValues[j,0])
                rates = net.createTopoInput(net.nE, net.popValues[j,i])
        rates += net.noise
        net.inputGroups[name+'e'].rate = rates 
        
        
def plotError(desiredResult, result, nE):
    error = (desiredResult - result)%1.
    correctionIdxs = np.where(error > 0.5)[0]
    correctedError = [1 - error[i] if (i in correctionIdxs) else error[i] for i in xrange(len(error))]
    correctedErrorSum = np.average(correctedError)
    print 'Inference error:', correctedErrorSum
        
    fi = plt.figure()#figsize = (5.0,4.6)
    ax = plt.subplot(1,1,1)
    matplotlib.rcParams.update({'font.size': 22})
    plt.scatter(desiredResult*nE, result*nE, c='k') #range(len(error))
    plt.title('Error: ' + str(correctedErrorSum))
    plt.xlabel('Desired activity')
    plt.ylabel('Population activity')
    ax.set_xticks([0,nE/2.,nE])     
    ax.set_xticklabels(['0', '800', '1600'])
    ax.set_yticks([0,nE/2.,nE])     
    ax.set_yticklabels(['0', '800', '1600'], va='center')
    plt.xlim(xmin = 0, xmax = nE)
    plt.ylim(ymin = 0, ymax = nE)
    return correctedErrorSum
        
        
def evaluate(dataPath, ending='1000', start_time=0, nE=1600, noise=''):
    end_time = int(ending)
    
    resultMonitor = np.loadtxt(dataPath + 'resultPopVecs' + ending + noise + '.txt')
    popVecs = np.loadtxt(dataPath + 'popVecs' + ending + noise + '.txt')
        
    result = resultMonitor[start_time:end_time,1]
    desiredResult = (resultMonitor[start_time:end_time,0])%1.
    correctedErrorSum = plotError(desiredResult, result, nE)
    plt.savefig(dataPath + 'evaluation_resultMonitorB' + ending + noise, dpi = 300)
    desiredResult = (popVecs[start_time:end_time,0])%1.
    correctedErrorSum = plotError(desiredResult, result, nE)
    plt.savefig(dataPath + 'evaluation_popVecsB' + ending + noise, dpi = 300)
    
    temp = np.zeros((1,1))
    temp[0,0] = correctedErrorSum
    np.savetxt(dataPath + 'errorB' + noise + '.txt', temp)
    
    result = resultMonitor[start_time:end_time,2]
    desiredResult = (resultMonitor[start_time:end_time,0])%1.
    correctedErrorSum = plotError(desiredResult, result, nE)
    plt.savefig(dataPath + 'evaluation_resultMonitorC' + ending + noise, dpi = 300)
    desiredResult = (popVecs[start_time:end_time,0])%1.
    correctedErrorSum = plotError(desiredResult, result, nE)
    plt.savefig(dataPath + 'evaluation_popVecsC' + ending + noise, dpi = 300)

    temp = np.zeros((1,1))
    temp[0,0] = correctedErrorSum
    np.savetxt(dataPath + 'errorC' + noise + '.txt', temp)





if __name__ == "__main__":
    evaluate('./', ending='10')






    
