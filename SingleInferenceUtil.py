import brian as b
from brian import *
import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import scipy 
import brian.experimental.realtime_monitor as rltmMon


def setNewInput(net,j):
    '''
        Assumes that the input net contains three input groups named X, Y, and Z which are stored in a dictionary inputGroups.
    '''
    for i,name in enumerate(net.inputPopulationNames):
        if name == 'X':
            net.popValues[j,i] = 0.5;
            rates = net.createTopoInput(net.nE, net.popValues[j,i])
            idxs = range(0,net.nE,net.numPeaks)
            rates[idxs] = 0
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

        
        
def plotActivity(dataPath, ax1=None):
    averagingWindowSize = 1
    nE = 1600
    ymax = 40
#     b.rcParams['lines.color'] = 'w'
#     b.rcParams['text.color'] = 'w'
#     b.rcParams['xtick.color'] = 'w'
#     b.rcParams['ytick.color'] = 'w'
#     b.rcParams['axes.labelcolor'] = 'w'
#     b.rcParams['axes.edgecolor'] = 'w'
    b.rcParams['font.size'] = 20
    if ax1==None:
        fig = b.figure(figsize=(8,6.5))
        fig_axis=b.subplot(1,1,1)
    else:
        fig_axis = ax1
        b.sca(ax1)
    path = dataPath + 'activity/'
    spikeCount = np.loadtxt(path + 'spikeCountCe.txt')
    popVecs = np.loadtxt(path + 'popVecs1.txt')
    desiredResult = (popVecs[0] + popVecs[1])%1.*1600
    resultMonitor = np.loadtxt(path + 'resultPopVecs1.txt')
    actualResult = resultMonitor[2]*1600
    ax1.axvline(desiredResult, color='r', linewidth=3, ymax=ymax, label='Desired result')
    ax1.axvline(actualResult, color='blue', linewidth=3, ymax=ymax, label='Population vector')
    
    print 'desiredResult', desiredResult, ', actual result', actualResult
#     spikeCount = np.roll(spikeCount, 800+int(-1*desiredResult))
    spikeCount = movingaverage(spikeCount,averagingWindowSize)
    b.plot(spikeCount, 'deepskyblue', alpha=0.6, linewidth=3, label='Population activity')
    fig_axis.set_xticks([0., nE/2, nE])
    fig_axis.set_xticklabels(['0', '0.5', '1'])
    fig_axis.set_yticks([0., ymax/2, ymax])
    fig_axis.spines['top'].set_visible(False)
    fig_axis.spines['right'].set_visible(False)
    fig_axis.get_xaxis().tick_bottom()
    fig_axis.get_yaxis().tick_left()
    b.ylabel('Firing Rate [Hz]')

    b.ylim(0,ymax)
    b.legend(fancybox=True, framealpha=0.0, loc='upper left')
#         b.title('spikes:' + str(sum(spikeCount)) + ', pop. value: ' + str(computePopVector(spikeCount)))
    if ax1==None:
        b.xlabel('Neuron number (resorted)')
        b.savefig(dataPath + 'SignalRestoration.png', dpi=900, transparent=True)
        
    
#     b.show()
    
if __name__ == "__main__":
    import os
    numPeaks = [50]
    plotActivity(os.getcwd()+'/SignalRestoration/', numPeaks)
    
    
    
