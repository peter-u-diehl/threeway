import brian as b
from brian import *
import numpy as np
import matplotlib
import matplotlib.cm as cmap
import scipy 
from scipy.optimize import curve_fit


def setNewInput(net, j):
    '''
        Assumes that the input net contains three input groups named X, Y, and Z which are stored in a dictionary inputGroups.
    '''
    for name in net.populationNames:        
        net.neuronGroups[name+'e'].v = net.v_restE
        net.neuronGroups[name+'i'].v = net.v_restI
        
    for i,name in enumerate(net.inputPopulationNames):
        if name == 'X':
            net.popValues[j,i] = (float(j)/net.numExamples+net.gaussianDist)%1.;
            rates = net.createTopoInput(net.nE, net.popValues[j,i])
            net.popValues[j,i] = (float(j)/net.numExamples-net.gaussianDist)%1.;
            rates = np.maximum(rates, net.createTopoInput(net.nE, net.popValues[j,i]) / net.gaussianPeak * net.gaussianPeakLow)
#             rates += net.createTopoInput(net.nE, net.popValues[j,i]) / net.gaussianPeak * net.gaussianPeakLow
#             print 'sum of inputs: ', sum(rates)
        else:
            rates = np.ones(net.nE)  * 0
        rates += net.noise
        net.inputGroups[name+'e'].rate = rates 
        
        
        
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

        
        
def plotActivity(dataPath, gaussianDistances, ax1=None):
    averagingWindowSize = 32
    
    if ax1==None:
        b.figure()
    else:
        b.sca(ax1)
        
    linewidth = 2
    for i,dist in enumerate(gaussianDistances[:]):
        path = dataPath + '/dist'+str(dist)+'/' +'activity/'
        spikeCountTemp = np.load(path + 'spikeCountPerExample.npy')
        spikeCount = spikeCountTemp[25,:,0]#np.loadtxt(path + 'spikeCountAe.txt')
#         spikeCount = np.roll(spikeCount, 400)
        inputSpikeCount = np.loadtxt(path + 'spikeCountXe.txt')
        
        spikeCount = movingaverage(spikeCount,averagingWindowSize)
        inputSpikeCount = movingaverage(inputSpikeCount,averagingWindowSize)
        if i==len(gaussianDistances)-1:
            b.plot(spikeCount, 'b', alpha=0.6, linewidth=linewidth, label='Max. dist output')#alpha=0.5+(0.5*float(i)/float(len(lower_peaks))), 
            b.plot(inputSpikeCount, 'r', alpha=1., linewidth=linewidth, label='Max. dist. input')
        elif i==0:
            b.plot(spikeCount, 'k--', alpha=0.6, linewidth=linewidth, label='Min. dist output')
            b.plot(inputSpikeCount, 'r--', alpha=1., linewidth=linewidth, label='Min. dist. input')
        else: 
            b.plot(spikeCount, 'k', alpha=0.2+(0.4*float(i)/float(len(gaussianDistances))), linewidth=0.6)
            b.plot(inputSpikeCount, 'r', alpha=0.2+(0.4*float(i)/float(len(gaussianDistances))), linewidth=0.6)
    b.ylim(0,35)

#         b.title('spikes:' + str(sum(spikeCount)) + ', pop. value: ' + str(computePopVector(spikeCount)))
    if ax1==None:
        b.legend(loc='upper left', fancybox=True, framealpha=0.0)
        b.savefig(dataPath + '/multipleAnglesActivity.png', dpi=300)
        
        
        
        
def plotSingleActivity(dataPath, gaussianDistance, ax1=None):
    b.rcParams['font.size'] = 20
    averagingWindowSize = 30
    nE = 1600
    ymax = 1
    
    if ax1==None:
        b.figure(figsize=(8,6.5))
        fig_axis = b.subplot(1,1,1)
    else:
        fig_axis = ax1
        b.sca(ax1)
        
    linewidth = 3
    path = dataPath + '/dist'+str(gaussianDistance)+'/' +'activity/'
    spikeCountTemp = np.load(path + 'spikeCountPerExample.npy')
    spikeCount = spikeCountTemp[25,:,0]#np.loadtxt(path + 'spikeCountAe.txt')
#     spikeCount = np.roll(spikeCount, 400)
    inputSpikeCount = np.roll(np.loadtxt(path + 'spikeCountXe.txt'), 400)
    
    spikeCount = movingaverage(spikeCount,averagingWindowSize)
    spikeCount /= np.max(spikeCount)
    inputSpikeCount = movingaverage(inputSpikeCount,averagingWindowSize)
    inputSpikeCount /= np.max(inputSpikeCount)
    b.plot(spikeCount, 'deepskyblue', alpha=0.6, linewidth=linewidth, label='Output')#alpha=0.5+(0.5*float(i)/float(len(lower_peaks))), 
    b.plot(inputSpikeCount, 'r', alpha=1., linewidth=linewidth, label='Input')
    fig_axis.set_xticks([0., nE/2, nE])
    fig_axis.set_xticklabels(['0', '0.5', '1'])
    fig_axis.set_yticks([0., ymax/2, ymax])
    fig_axis.spines['top'].set_visible(False)
    fig_axis.spines['right'].set_visible(False)
    fig_axis.get_xaxis().tick_bottom()
    fig_axis.get_yaxis().tick_left()
    b.ylabel('Normalized response')
    b.ylim(0,ymax)

#         b.title('spikes:' + str(sum(spikeCount)) + ', pop. value: ' + str(computePopVector(spikeCount)))
    if ax1==None:
        b.legend(loc='upper left', fancybox=True, framealpha=0.0)
        b.savefig(dataPath + '/multipleAnglesSingleActivity.png', dpi=300)
        
        
        
        
def plotHeatmap(dataPath, gaussianDistances, ax1=None):
    b.rcParams['font.size'] = 20
    if ax1==None:
        b.figure(figsize=(8,15))
    else:
        b.sca(ax1)
        
    nE = 1600
    averagingWindowSize = 32
    
    spikeCount = np.zeros((len(gaussianDistances), nE))
    for i,dist in enumerate(gaussianDistances[:]):
        path = dataPath + '/dist'+str(dist)+'/' +'activity/'
        spikeCountTemp = np.load(path + 'spikeCountPerExample.npy')
        spikeCount[i,:] = spikeCountTemp[25,:,0]#np.loadtxt(path + 'spikeCountAe.txt')
#         spikeCount[i,:] = np.roll(spikeCount[i,:], int(0.25*len(spikeCount[i,:])))
        spikeCount[i,:] = movingaverage(spikeCount[i,:], averagingWindowSize)
        spikeCount[i,:] /= np.max(spikeCount[i,:])
    
    b.imshow(spikeCount[:,:], aspect='auto', extent=[0,1,2,0])
    b.colorbar()
    b.xlabel('Neuron number (resorted)')
    b.xlabel('Neuron number (resorted)')
    
    
    if ax1==None:
        b.savefig(dataPath + '/multipleAnglesHeatmap.png', dpi=300, bbox_inches='tight')
    
        
def plotSingleTuningCurve(dataPath, gaussianDistance, ax1=None):
    b.rcParams['font.size'] = 20
    averagingWindowSize = 1
    nE = 1600
    ymax = 1
    
    if ax1==None:
        b.figure(figsize=(8,6.5))
        fig_axis = b.subplot(1,1,1)
    else:
        fig_axis = ax1
        b.sca(ax1)
        
    path = dataPath + '/dist'+str(gaussianDistance)+'/' +'activity/'
    spikeCount = np.load(path + 'spikeCountPerExample.npy')
    spikeCountSingleNeuron = (spikeCount[:,nE/2-2,0])
    numMeasurements = len(spikeCount[:,0,0])
    
    spikeCount = movingaverage(spikeCountSingleNeuron,averagingWindowSize)
    spikeCount /= np.max(spikeCountSingleNeuron)
    b.plot(spikeCount, color='deepskyblue', marker='o', alpha=0.6, linewidth=0, label='Output')#alpha=0.5+(0.5*float(i)/float(len(lower_peaks))), 
    fig_axis.set_xticks([0., numMeasurements/2, numMeasurements])
    fig_axis.set_xticklabels(['0', '0.5', '1'])
    fig_axis.set_yticks([0., ymax/2, ymax])
    fig_axis.spines['top'].set_visible(False)
    fig_axis.spines['right'].set_visible(False)
    fig_axis.get_xaxis().tick_bottom()
    fig_axis.get_yaxis().tick_left()
    b.ylabel('Normalized response')
    b.ylim(0,ymax+0.05)

#         b.title('spikes:' + str(sum(spikeCount)) + ', pop. value: ' + str(computePopVector(spikeCount)))
    if ax1==None:
        b.legend(loc='upper left', fancybox=True, framealpha=0.0)
        b.savefig(dataPath + '/tuningCurveSingleNeuron.png', dpi=300, bbox_inches='tight')
        
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
        
def plotPopulationTuningCurve(dataPath, gaussianDistance, ax1=None):
    b.rcParams['font.size'] = 20
    averagingWindowSize = 1
    nE = 1600
    ymax = 1
    
    if ax1==None:
        b.figure(figsize=(8,6.5))
        fig_axis = b.subplot(1,1,1)
    else:
        fig_axis = ax1
        b.sca(ax1)
        
    
    path = dataPath + '/dist'+str(gaussianDistance)+'/' +'activity/'
    spikeCount = np.load(path + 'spikeCountPerExample.npy')
    numMeasurements = len(spikeCount[:,0,0])
    measurementSpace = np.arange(numMeasurements)
    populationSpikeCount = np.zeros((numMeasurements))
    for i in xrange(nE):
        populationSpikeCount += np.roll(spikeCount[:,i,0], int(-1.*i/nE*numMeasurements+ numMeasurements/2))
    populationSpikeCount = movingaverage(populationSpikeCount,averagingWindowSize)
    populationSpikeCount /= np.max(populationSpikeCount)
    if gaussianDistance==0.:
        mean = sum(measurementSpace*populationSpikeCount)/numMeasurements                   #note this correction
        sigma = sum(populationSpikeCount*(measurementSpace-mean)**2)/numMeasurements        #note this correction
        popt, _ = curve_fit(gaus,measurementSpace,populationSpikeCount,p0=[1,mean,sigma])
        print 'Gaussian amplitude: ', popt[0], ', mean: ', popt[1], ', std.: ', popt[2]**2
        
        b.plot(measurementSpace,gaus(measurementSpace,*popt),'k')
    
    b.plot(populationSpikeCount, color='deepskyblue', marker='o', alpha=0.6, linewidth=0, label='Output')#alpha=0.5+(0.5*float(i)/float(len(lower_peaks))), 
    fig_axis.set_xticks([0., numMeasurements/2, numMeasurements])
    fig_axis.set_xticklabels(['0', '0.5', '1'])
    fig_axis.set_yticks([0., ymax/2, ymax])
    fig_axis.spines['top'].set_visible(False)
    fig_axis.spines['right'].set_visible(False)
    fig_axis.get_xaxis().tick_bottom()
    fig_axis.get_yaxis().tick_left()
#     b.ylabel('Normalized response')
    b.ylim(0,ymax+0.05)

#         b.title('spikes:' + str(sum(spikeCount)) + ', pop. value: ' + str(computePopVector(spikeCount)))
    if ax1==None:
        b.legend(loc='upper left', fancybox=True, framealpha=0.0)
        b.savefig(dataPath + '/tuningCurvePopulation' + str(gaussianDistance) + '.png', dpi=300, bbox_inches='tight')
        
        
def plotTuningHeatmap(dataPath, gaussianDistances, ax1=None):
    b.rcParams['font.size'] = 20
#     averagingWindowSize = 1
    nE = 1600
    if ax1==None:
        b.figure(figsize=(8,15))
        fig_axis = b.subplot(1,1,1)
    else:
        b.sca(ax1)
        fig_axis=ax1
    path = dataPath + '/dist'+str(gaussianDistances[0])+'/' +'activity/'
    spikeCount = np.load(path + 'spikeCountPerExample.npy')
    numMeasurements = len(spikeCount[:,0,0])
    measurementSpace = np.arange(numMeasurements)
    populationSpikeCount = np.zeros((numMeasurements, len(gaussianDistances)))
        
    for i,gaussianDistance in enumerate(gaussianDistances):
#         distanceAngle = gaussianDistance * 2 * 360
        path = dataPath + '/dist'+str(gaussianDistance)+'/' +'activity/'
        spikeCount = np.load(path + 'spikeCountPerExample.npy')
        for j in xrange(nE):
            populationSpikeCount[:,i] += np.roll(spikeCount[:,j,0], int(-1.*j/nE*numMeasurements+ numMeasurements/2))
#         j = 0
#         populationSpikeCount[:,i] += np.roll(spikeCount[:,j,0], int(-1.*j/nE*numMeasurements+ numMeasurements/2))
#         populationSpikeCount[:,i] = movingaverage(populationSpikeCount[:,i],averagingWindowSize)
        populationSpikeCount[:,i] /= np.max(populationSpikeCount[:,i])
        
        if gaussianDistance==0.:
            mean = sum(measurementSpace*populationSpikeCount[:,i])/numMeasurements                   #note this correction
            sigma = sum(populationSpikeCount[:,i]*(measurementSpace-mean)**2)/numMeasurements        #note this correction
            popt, _ = curve_fit(gaus,measurementSpace,populationSpikeCount[:,i],p0=[1,mean,sigma])
#             tuningWidth = abs(popt[2])*2
            tuningWidth = np.sum(populationSpikeCount>=0.5)
            tuningWidthAngle = tuningWidth/50.*360.
            print 'Gaussian amplitude:', popt[0], 'mean:', popt[1], 'std.:', popt[2], \
                ', tuning width:', tuningWidth, ', tuning width angle:', tuningWidthAngle
        
        
        
    minX = max(0,round(numMeasurements/2-tuningWidth*2))
    maxX = min(numMeasurements, round(numMeasurements/2+tuningWidth*2))
    minY = round(0)
    maxY = round(tuningWidth/50.*len(gaussianDistances)*2*2)
    print minX, maxX, minY, maxY
    croppedCount = populationSpikeCount[minX:maxX,minY:maxY]
    fig_axis.imshow(croppedCount.transpose(), aspect='auto', 
                    extent=[minX, maxX, 2, 0])
#     b.colorbar()
    fig_axis.set_xlabel('Distance from preferred stimulus, \ndivided by tuning width')
    fig_axis.set_xticks([numMeasurements/2-tuningWidth*2, numMeasurements/2-tuningWidth, 
                         numMeasurements/2, numMeasurements/2+tuningWidth, numMeasurements/2+tuningWidth*2])
    fig_axis.set_xticklabels(['-2', '-1', '0', '1', '2'])
    fig_axis.set_yticks([0., 0.5, 1, 1.5, 2])
    fig_axis.spines['top'].set_visible(False)
    fig_axis.spines['right'].set_visible(False)
    fig_axis.spines['bottom'].set_visible(False)
    fig_axis.spines['left'].set_visible(False)
    fig_axis.get_xaxis().tick_bottom()
    fig_axis.get_yaxis().tick_left()
    
    if ax1==None:
        b.savefig(dataPath + '/tuningHeatmap.png', dpi=300, bbox_inches='tight')
    
    return tuningWidthAngle
    
if __name__ == "__main__":
    import os
#     plotActivity(os.getcwd()+'/multipleAngles', np.linspace(0,0.25,13))
#     plotSingleActivity(os.getcwd()+'/multipleAngles', 0.125)
#     plotHeatmap(os.getcwd()+'/multipleAngles', np.linspace(0,0.25,13))
#     plotSingleTuningCurve(os.getcwd()+'/multipleAngles', 0.125)
#     plotPopulationTuningCurve(os.getcwd()+'/multipleAngles', 0.0833333333333)
    plotTuningHeatmap(os.getcwd()+'/multipleAngles', np.linspace(0,0.25,13))
    
    #  0deg 0.0
    # 30deg 0.0416666
    # 60deg 0.0833333333333
    # 90deg 0.125
    #120deg 0.16666667
    #150deg 0.20833333
    #180deg 0.25 
    
