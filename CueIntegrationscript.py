import os
import threeWay


if __name__ == "__main__":
    argDict = threeWay.getArgs()
    print argDict

    numTestingExamples = 1
    testingNoise = 0
    argDict['numExamples'] = numTestingExamples
    argDict['testMode'] = True
    argDict['weightPath'] = os.getcwd()+'/sortedWeights/'
    argDict['noise'] = testingNoise
    argDict['inputType'] = 'CueIntegration'
    argDict['inputWeight'] = 0.12
    argDict['recurrentWeight'] = 0.001
    argDict['singleExampleTime'] = 60.0
    
    lower_peaks = [11]
    for peak in lower_peaks:
        argDict['targetPath'] = os.getcwd()+'/CueIntegration/peak_'+str(peak)+'/'
        argDict['gaussianPeakLow'] = peak
        relNet = threeWay.RelationalNetwork(**argDict)
        relNet.createNetwork()
        relNet.run()
        del relNet
        
    activityDataPath = os.getcwd()+'/CueIntegration/'
    import CueIntegrationUtil
    CueIntegrationUtil.plotActivity(activityDataPath, lower_peaks)
    


