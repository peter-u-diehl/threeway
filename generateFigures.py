import os
import threeWay
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.colors import transparent



def cueIntegration(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'CueIntegration'
    lower_peaks = [11]
    if runSimulations:
        for peak in lower_peaks:
            argDict['targetPath'] = os.getcwd()+'/CueIntegration/peak_'+str(peak)+'/'
            argDict['gaussianPeakLow'] = peak
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/CueIntegration/'
    import CueIntegrationUtil
    CueIntegrationUtil.plotActivity(activityDataPath, lower_peaks, ax1=ax1)


def singleCueIntegration(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'CueIntegration'
    lower_peaks = [11]
    if runSimulations:
        for peak in lower_peaks:
            argDict['targetPath'] = os.getcwd()+'/CueIntegration/peak_'+str(peak)+'/'
            argDict['gaussianPeakLow'] = peak
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/CueIntegration/'
    import SingleCueIntegrationUtil
    SingleCueIntegrationUtil.plotActivity(activityDataPath, lower_peaks, ax1=ax1)



def singleSignalRestoration(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'SignalRestoration'
    numPeaks = [50]
    if runSimulations:
        for peaks in numPeaks:
            argDict['targetPath'] = os.getcwd()+'/SignalRestoration/peak_'+str(peaks)+'/'
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.numPeaks = peaks
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/SignalRestoration/'
    import SingleSignalRestorationUtil
    SingleSignalRestorationUtil.plotActivity(activityDataPath, numPeaks, ax1=ax1)
    
    
    
def WTA(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'WTA'
    lower_peaks = [11]
    if runSimulations:
        for peak in lower_peaks:
            argDict['targetPath'] = os.getcwd()+'/WTA/peak_'+str(peak)+'/'
            argDict['gaussianPeakLow'] = peak
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/WTA/'
    import WTAUtil
    WTAUtil.plotActivity(activityDataPath, lower_peaks, ax1=ax1)
    
    
    
def singleWTA(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'WTA'
    lower_peaks = [11]
    if runSimulations:
        for peak in lower_peaks:
            argDict['targetPath'] = os.getcwd()+'/WTA/peak_'+str(peak)+'/'
            argDict['gaussianPeakLow'] = peak
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/WTA/'
    import SingleWTAUtil
    SingleWTAUtil.plotActivity(activityDataPath, lower_peaks, ax1=ax1)
    


def inputOutputBeforeLearning(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'InputOutput'
    inputStrengths = np.logspace(-0.5, 2.3, 20, base=10)
    if runSimulations:
        for inputStrength in inputStrengths:
            argDict['targetPath'] = os.getcwd()+'/InputOutputBeforeLearning/in_'+str(inputStrength)+'/'
            argDict['gaussianPeak'] = inputStrength
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/InputOutputBeforeLearning/'
    import InputOutputBeforeLearningUtil
    InputOutputBeforeLearningUtil.plotActivity(activityDataPath, inputStrengths, ax1=ax1)
    


def singleInputOutputBeforeLearning(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'InputOutput'
    inputStrengths = [20]
    if runSimulations:
        for inputStrength in inputStrengths:
            argDict['targetPath'] = os.getcwd()+'/InputOutputBeforeLearning/in_'+str(inputStrength)+'/'
            argDict['gaussianPeak'] = inputStrength
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/InputOutputBeforeLearning/'
    import SingleInputOutputBeforeLearningUtil
    SingleInputOutputBeforeLearningUtil.plotActivity(activityDataPath, inputStrengths, ax1=ax1)
    


def inputOutput(argDict, ax1=None, fig=None, runSimulations=True):
    argDict['inputType'] = 'InputOutput'
    inputStrengths = np.logspace(-0.5, 2.3, 20, base=10)
    if runSimulations:
        for inputStrength in inputStrengths:
            argDict['targetPath'] = os.getcwd()+'/InputOutput/in_'+str(inputStrength)+'/'
            argDict['gaussianPeak'] = inputStrength
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/InputOutput/'
    import InputOutputUtil
    InputOutputUtil.plotActivity(activityDataPath, inputStrengths, ax1=ax1, fig=fig)
    


def singleInputOutput(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'InputOutput'
    inputStrengths = [20]
    if runSimulations:
        for inputStrength in inputStrengths:
            argDict['targetPath'] = os.getcwd()+'/InputOutput/in_'+str(inputStrength)+'/'
            argDict['gaussianPeak'] = inputStrength
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/InputOutput/'
    import SingleInputOutputUtil
    SingleInputOutputUtil.plotActivity(activityDataPath, inputStrengths, ax1=ax1)
    
    

def contrastInvariance(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'InputOutput'
    inputStrengths = [10, 30, 100] #20,  5, 
    if runSimulations:
        for inputStrength in inputStrengths:
            argDict['targetPath'] = os.getcwd()+'/ContrastInvariance/in_'+str(inputStrength)+'/'
            argDict['gaussianPeak'] = inputStrength
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/ContrastInvariance/'
    import ContrastInvarianceUtil
    ContrastInvarianceUtil.plotActivity(activityDataPath, inputStrengths, ax1=ax1)
    
    
def multipleAngles(argDict, ax1=None, ax2=None, ax3=None, ax4=None, ax5=None, ax6=None, runSimulations=True):
    argDict['inputType'] = 'multipleAngles'
    argDict['gaussianPeak'] = 20
    argDict['gaussianPeakLow'] = 20
    argDict['gaussianSigma'] = 1./8.
    numRuns = 4   #61
    gaussianDistances = np.linspace(0,0.25,numRuns)
    if runSimulations:
        argDict['numExamples'] = 50
        for dist in gaussianDistances:
            argDict['targetPath'] = os.getcwd()+'/multipleAngles/dist' + str(dist) + '/'
            argDict['gaussianDist'] = dist
            relNet = threeWay.RelationalNetwork(**argDict)
            relNet.createNetwork()
            relNet.run()
            del relNet
        
    activityDataPath = os.getcwd()+'/multipleAngles/'
    import MultipleAnglesUtil
    tuningWidthAngle = MultipleAnglesUtil.plotTuningHeatmap(activityDataPath, gaussianDistances, ax1=ax1)
    tuningWidth = tuningWidthAngle/180. # the distance is between 0 and 180 degrees
    print tuningWidth, tuningWidthAngle
    if round((numRuns-1)*tuningWidth*2) < 0.25:
        MultipleAnglesUtil.plotPopulationTuningCurve(activityDataPath, 
                                                     gaussianDistances[round((numRuns-1)/6.*0)], ax1=ax2)
        MultipleAnglesUtil.plotPopulationTuningCurve(activityDataPath, 
                                                     gaussianDistances[round((numRuns-1)*tuningWidth*0.5)], ax1=ax3)
        MultipleAnglesUtil.plotPopulationTuningCurve(activityDataPath, 
                                                     gaussianDistances[round((numRuns-1)*tuningWidth*1)], ax1=ax4)
        MultipleAnglesUtil.plotPopulationTuningCurve(activityDataPath, 
                                                     gaussianDistances[round((numRuns-1)*tuningWidth*1.5)], ax1=ax5)
        MultipleAnglesUtil.plotPopulationTuningCurve(activityDataPath,
                                                      gaussianDistances[round((numRuns-1)*tuningWidth*2)], ax1=ax6)
    else:
        print 'Generating multiple angles figure failed due to too late split of activity. Possibly the network was not trained correctly.'
        

def singleInference(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'add'
    argDict['seed'] = 1
    if runSimulations:
        argDict['targetPath'] = os.getcwd()+'/inference/singleInference/'
        relNet = threeWay.RelationalNetwork(**argDict)
        relNet.createNetwork()
        relNet.run()
        del relNet
        
    activityDataPath = os.getcwd()+'/inference/singleInference/'
    import SingleInferenceUtil
    SingleInferenceUtil.plotActivity(activityDataPath, ax1=ax1)




def testInference(argDict, ax1=None, runSimulations=True):
    argDict['inputType'] = 'add'
    argDict['seed'] = 1
    argDict['numExamples'] = 1000
    if runSimulations:
        argDict['targetPath'] = os.getcwd()+'/inference/testInference/'
        relNet = threeWay.RelationalNetwork(**argDict)
        relNet.createNetwork()
        relNet.run()
        del relNet
        
    activityDataPath = os.getcwd()+'/inference/testInference/'
    import testInferenceUtil
    testInferenceUtil.plotActivity(activityDataPath, ax1=ax1)



def generateFig1(argDict, runSimulations, xPosLabel, yPosLabel, xPosLabel2, yPosLabel2):
    unused_fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(5, figsize=(12,24), sharex=True)
    weightPath = argDict['weightPath']
    argDict['weightPath'] = os.getcwd()+'/random/'
    singleInputOutputBeforeLearning(argDict, ax1=ax1, runSimulations=runSimulations)
    t1 = ax1.text(xPosLabel, yPosLabel, 'A', fontsize=35, fontweight='bold', 
            transform=ax1.transAxes)
    t6 = ax1.text(xPosLabel2, yPosLabel2, 'Before learning', fontsize=25, #fontweight='bold', 
            horizontalalignment='right',
            transform=ax1.transAxes)
    
    argDict['weightPath'] = weightPath
    singleInputOutput(argDict, ax1=ax2, runSimulations=runSimulations)
    t2 = ax2.text(xPosLabel, yPosLabel, 'B', fontsize=35, fontweight='bold', 
            transform=ax2.transAxes)
    t7 = ax2.text(xPosLabel2, yPosLabel2, 'After learning', fontsize=25, #fontweight='bold', 
            horizontalalignment='right',
            transform=ax2.transAxes)
    
    singleSignalRestoration(argDict, ax1=ax3, runSimulations=True)
    t3 = ax3.text(xPosLabel, yPosLabel, 'C', fontsize=35, fontweight='bold', 
            transform=ax3.transAxes)
    t8 = ax3.text(xPosLabel2, yPosLabel2, 'Signal restoration', fontsize=25, #fontweight='bold', 
            horizontalalignment='right',
            transform=ax3.transAxes)
    
    singleCueIntegration(argDict, ax1=ax4, runSimulations=runSimulations)
    t4 = ax4.text(xPosLabel, yPosLabel, 'D', fontsize=35, fontweight='bold', 
            transform=ax4.transAxes)
    t9 = ax4.text(xPosLabel2, yPosLabel2, 'Cue integration', fontsize=25, #fontweight='bold', 
            horizontalalignment='right',
            transform=ax4.transAxes)
    
    singleWTA(argDict, ax1=ax5, runSimulations=runSimulations)
    t5 = ax5.text(xPosLabel, yPosLabel, 'E', fontsize=35, fontweight='bold', 
            transform=ax5.transAxes)
    t10 = ax5.text(xPosLabel2, yPosLabel2, 'soft winner-take-all', fontsize=25, #fontweight='bold', 
            horizontalalignment='right',
            transform=ax5.transAxes)
#     WTA(argDict, ax1=ax5, runSimulations=runSimulations)
    ax5.set_xlabel('Neuron number (resorted)')
#     fig.tight_layout()
    plt.savefig('responsePatterns', transparent=False, bbox_inches='tight', 
                bbox_extra_artists=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10])
    

def generateFig2(argDict, runSimulations, xPosLabel, yPosLabel, xPosLabel2, yPosLabel2):
    fig, ((ax1, ax2)) = plt.subplots(2, figsize=(12,24))
    # inputOutputBeforeLearning(argDict, ax1=ax1, runSimulations=runSimulations)
    # fig.tight_layout()
    # plt.savefig('inputOutputBeforeLearning', transparent=False, bbox_inches='tight')
    inputOutput(argDict, ax1=ax1, fig=fig, runSimulations=runSimulations)
    contrastInvariance(argDict, ax1=ax2, runSimulations= runSimulations)
    # fig.tight_layout()
    plt.savefig('inputOutputRelation', transparent=False, bbox_inches='tight')


def generateFig3(argDict, runSimulations, xPosLabel, yPosLabel, xPosLabel2, yPosLabel2):
    fig = plt.figure(figsize=(12,15))
    ax = plt.subplot(111)
    numX = 5
    numY = 5
    colspanTuning = 2
    ax1 = plt.subplot2grid((numY,numX), (0,colspanTuning), colspan=numX-colspanTuning, rowspan=numY)
    ax2 = plt.subplot2grid((numY,numX), (0,0), colspan=colspanTuning)
    ax3 = plt.subplot2grid((numY,numX), (1,0), colspan=colspanTuning)
    ax4 = plt.subplot2grid((numY,numX), (2,0), colspan=colspanTuning)
    ax5 = plt.subplot2grid((numY,numX), (3,0), colspan=colspanTuning)
    ax6 = plt.subplot2grid((numY,numX), (4,0), colspan=colspanTuning)
    t1 = ax2.text(xPosLabel2, yPosLabel2, 'Single input', fontsize=18, #fontweight='bold', 
            horizontalalignment='right', transform=ax2.transAxes)
    t2 = ax3.text(xPosLabel2, yPosLabel2, '0.5 tuning-width\ndistance', fontsize=18, #fontweight='bold', 
            horizontalalignment='right', transform=ax3.transAxes)
    t3 = ax4.text(xPosLabel2, yPosLabel2, '1 tuning-width\ndistance', fontsize=18, #fontweight='bold', 
            horizontalalignment='right', transform=ax4.transAxes)
    t4 = ax5.text(xPosLabel2, yPosLabel2, '1.5 tuning-width\ndistance', fontsize=18, #fontweight='bold', 
            horizontalalignment='right', transform=ax5.transAxes)
    t5 = ax6.text(xPosLabel2, yPosLabel2, '2.0 tuning-width\ndistance', fontsize=18, #fontweight='bold', 
            horizontalalignment='right', transform=ax6.transAxes)
    ax.set_ylabel('Angle between inputs, divided by tuning width')
    ax6.set_xlabel('Preferred stimulus')
    multipleAngles(argDict, ax1=ax1, ax2=ax2, ax3=ax3, ax4=ax4, ax5=ax5, ax6=ax6, runSimulations=runSimulations)
    fig.tight_layout()
    plt.savefig('multipleAngles', transparent=False, #bbox_inches='tight', 
                bbox_extra_artists=[t1, t2, t3, t4, t5,ax])#

def generateFig4(argDict, runSimulations, xPosLabel, yPosLabel, xPosLabel2, yPosLabel2):
    plt.figure(figsize=(15,10))
    ax1 = plt.subplot(111)
    singleInference(argDict, ax1, runSimulations=runSimulations)
    ax1.set_xlabel('Neuron number (resorted)')
    
    plt.savefig('singleInference', transparent=False, bbox_inches='tight')



def generateFig5(argDict, runSimulations, xPosLabel, yPosLabel, xPosLabel2, yPosLabel2):
    plt.figure(figsize=(30,10))
    ax1 = plt.subplot(121)
    t1 = ax1.text(xPosLabel, yPosLabel, 'A', fontsize=35, fontweight='bold', 
            transform=ax1.transAxes)
    singleInference(argDict, ax1, runSimulations=runSimulations)
    ax1.set_xlabel('Neuron number (resorted)')

    ax2 = plt.subplot(122)
    t2 = ax2.text(xPosLabel+1.17, yPosLabel, 'B', fontsize=35, fontweight='bold', 
            transform=ax1.transAxes)
    testInference(argDict, ax2, runSimulations=runSimulations)
    
    plt.savefig('inference', transparent=False, bbox_inches='tight', 
                bbox_extra_artists=[t1, t2])




def generateAllFigs(argDict):
    runSimulations = True
    numTestingExamples = 1
    singleExampleTime = 0.1 #in seconds
    import copy
    startingDict = copy.deepcopy(argDict)

    xPosLabel = -0.16
    yPosLabel = 1.0
    xPosLabel2 = 0.99
    yPosLabel2 = 0.8
    
    figGens = [
#                     generateFig1,
#                     generateFig2,
#                     generateFig3,
                    generateFig4,
                    generateFig5,
               ]
    for figGenerator in figGens:
        argDict = startingDict
        print 'fig gen', argDict['weightPath']
        if 'weightPath' not in argDict:
            argDict['weightPath'] = argDict['dataPath']+'/sortedWeights/'
        argDict['numExamples'] = numTestingExamples
        argDict['testMode'] = True
        argDict['singleExampleTime'] = singleExampleTime
        argDict['seed'] = 0
        figGenerator(argDict, runSimulations, xPosLabel, yPosLabel, xPosLabel2, yPosLabel2)

#     plt.show()


if __name__ == "__main__":
    argDict = threeWay.getArgs()
    print argDict
    generateAllFigs(argDict)











