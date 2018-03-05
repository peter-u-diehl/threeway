import brian as b
from brian import *
import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import scipy 
import brian.experimental.realtime_monitor as rltmMon


def setNewInput(net, j):
    '''
        Assumes that the input net contains three input groups named X, Y, and Z which are stored in a dictionary inputGroups.
    '''
    for i,name in enumerate(net.inputPopulationNames):
        if name == 'X':
            net.popValues[j,i] = 0.5;
            rates = net.createTopoInput(net.nE, net.popValues[j,i])
            net.popValues[j,i] = 0.;
            rates += net.createTopoInput(net.nE, net.popValues[j,i]) / net.gaussianPeak * net.gaussianPeakLow
            print 'sum of inputs: ', sum(rates)
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
        
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

        
        
def plotActivity(dataPath, lower_peaks, ax1=None):
    averagingWindowSize = 32
    
    if ax1==None:
        b.figure()
    else:
        b.sca(ax1)
        
    for i,gaussian_peak_low in enumerate(lower_peaks[:]):
        path = dataPath + '/peak_'+str(gaussian_peak_low)+'/' +'activity/'
        spikeCount = np.loadtxt(path + 'spikeCountAe.txt')
        inputSpikeCount = np.loadtxt(path + 'spikeCountXe.txt')
        
        spikeCount = movingaverage(spikeCount,averagingWindowSize)
        inputSpikeCount = movingaverage(inputSpikeCount,averagingWindowSize)
        if i==len(lower_peaks)-1:
            b.plot(spikeCount, 'b', alpha=1., linewidth=3, label='Output')#alpha=0.5+(0.5*float(i)/float(len(lower_peaks))), 
            b.plot(inputSpikeCount, 'r', alpha=1., linewidth=3, label='Input')
        elif i==0:
            b.plot(spikeCount, 'k--', alpha=1., linewidth=3)#
            b.plot(inputSpikeCount, 'r--', alpha=1., linewidth=3)  
        else: 
            b.plot(spikeCount, 'k', alpha=0.2+(0.4*float(i)/float(len(lower_peaks))), linewidth=0.6)
            b.plot(inputSpikeCount, 'r', alpha=0.2+(0.4*float(i)/float(len(lower_peaks))), linewidth=0.6)

    b.legend()
    b.ylim(0,35)
#         b.title('spikes:' + str(sum(spikeCount)) + ', pop. value: ' + str(computePopVector(spikeCount)))
    if ax1==None:
        b.savefig(dataPath + 'WTA.png', dpi=900)
        
    
#     b.show()
    
if __name__ == "__main__":
    import os
    plotActivity(os.getcwd()+'/WTA/activity/', [0,1,2,5,8,10,12,15,20])
    
    
    
