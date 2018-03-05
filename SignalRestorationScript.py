import os
import threeWay
import numpy as np


if __name__ == "__main__":
    argDict = threeWay.getArgs()
    print argDict

    numTestingExamples = 1
    testingNoise = 0
    argDict['numExamples'] = numTestingExamples
    argDict['testMode'] = True
    argDict['weightPath'] = os.getcwd()+'/sortedWeights/'
    argDict['noise'] = testingNoise
    argDict['inputType'] = 'SignalRestoration'
    argDict['inputWeight'] = 0.12
    argDict['recurrentWeight'] = 0.001
    argDict['singleExampleTime'] = 60.0
    
    numPeaks = [50]
    for peaks in numPeaks:
        argDict['targetPath'] = os.getcwd()+'/SignalRestoration/peak_'+str(peaks)+'/'
        relNet = threeWay.RelationalNetwork(**argDict)
        relNet.numPeaks = peaks
        relNet.createNetwork()
        relNet.run()
        del relNet
        
    activityDataPath = os.getcwd()+'/SignalRestoration/'
    import SignalRestorationUtil
    SignalRestorationUtil.plotActivity(activityDataPath, numPeaks)
    


