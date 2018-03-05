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

        
        
def plotActivity(dataPath, inputStrengths, ax1=None):
    averagingWindowSize = 1
    nE = 1600
    ymax = 30
#     b.rcParams['lines.color'] = 'w'
#     b.rcParams['text.color'] = 'w'
#     b.rcParams['xtick.color'] = 'w'
#     b.rcParams['ytick.color'] = 'w'
#     b.rcParams['axes.labelcolor'] = 'w'
#     b.rcParams['axes.edgecolor'] = 'w'
    b.rcParams['font.size'] = 20
    
    if ax1==None:
        b.figure(figsize=(8,6.5))
        fig_axis = b.subplot(1,1,1)
    else:
        fig_axis = ax1
        b.sca(ax1)
    for i,inputStrength in enumerate(inputStrengths[:]):
        path = dataPath + '/peak_'+str(inputStrength)+'/' +'activity/'
        spikeCount = np.loadtxt(path + 'spikeCountAe.txt')
        inputSpikeCount = np.loadtxt(path + 'spikeCountXe.txt')
        
        spikeCount = movingaverage(spikeCount,averagingWindowSize)
        inputSpikeCount = movingaverage(inputSpikeCount,averagingWindowSize)
        if i==len(inputStrengths)-1:
            b.plot(inputSpikeCount, 'r', alpha=1., linewidth=3, label='Input      ')
#         elif i==0:
#             b.plot(spikeCount, 'k--', alpha=1., linewidth=3)#
#             b.plot(inputSpikeCount, 'r--', alpha=1., linewidth=3)  
        else: 
            b.plot(inputSpikeCount, 'r', alpha=0.2+(0.4*float(i)/float(len(inputStrength))), linewidth=0.6)
    fig_axis.set_xticks([0., nE/2, nE])
    fig_axis.set_yticks([0., ymax/2, ymax])
    fig_axis.spines['top'].set_visible(False)
    fig_axis.spines['right'].set_visible(False)
    fig_axis.get_xaxis().tick_bottom()
    fig_axis.get_yaxis().tick_left()
    b.xlabel('Neuron number (resorted)')
    b.ylabel('Firing Rate [Hz]')

    b.legend(loc='upper left', fancybox=True, framealpha=0.0)
    b.ylim(0,ymax)
#         b.title('spikes:' + str(sum(spikeCount)) + ', pop. value: ' + str(computePopVector(spikeCount)))
    if ax1==None:
        b.savefig(dataPath + 'Normal_input.png', dpi=900, transparent=True)
        
    
    for i,inputStrength in enumerate(inputStrengths[:]):
        path = dataPath + '/peak_'+str(inputStrength)+'/' +'activity/'
        spikeCount = np.loadtxt(path + 'spikeCountAe.txt')
        inputSpikeCount = np.loadtxt(path + 'spikeCountXe.txt')
        
        spikeCount = movingaverage(spikeCount,averagingWindowSize)
        inputSpikeCount = movingaverage(inputSpikeCount,averagingWindowSize)
        if i==len(inputStrengths)-1:
            b.plot(spikeCount, 'deepskyblue', alpha=0.6, linewidth=3, label='Output')#alpha=0.5+(0.5*float(i)/float(len(numPeaks))), 
#         elif i==0:
#             b.plot(spikeCount, 'k--', alpha=1., linewidth=3)#
#             b.plot(inputSpikeCount, 'r--', alpha=1., linewidth=3)  
        else: 
            b.plot(spikeCount, 'k', alpha=0.2+(0.4*float(i)/float(len(inputStrength))), linewidth=0.6)
    b.legend(loc='upper left', fancybox=True, framealpha=0.0)
    fig_axis.set_xticks([0., nE/2, nE])
    fig_axis.set_yticks([0., ymax/2, ymax])
    fig_axis.spines['top'].set_visible(False)
    fig_axis.spines['right'].set_visible(False)
    fig_axis.get_xaxis().tick_bottom()
    fig_axis.get_yaxis().tick_left()
    b.xlabel('Neuron number (resorted)')
    b.ylabel('Firing Rate [Hz]')
    b.ylim(0,ymax)

    if ax1==None:
        b.savefig(dataPath + 'Normal.png', dpi=900, transparent=True)
#     b.show()
    
if __name__ == "__main__":
    import os
    inputStrength = [0]
    plotActivity(os.getcwd()+'/WTA/', inputStrength)
    
    
    
