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
    argDict['weightPath'] = os.getcwd()+'/random/'
    argDict['noise'] = testingNoise
    argDict['inputType'] = 'InputOutput'
    argDict['inputWeight'] = 0.12
    argDict['recurrentWeight'] = 0.001
    argDict['singleExampleTime'] = 60.0
    
    inputStrengths = np.logspace(-0.5, 2.3, 20, base=10)
    inputStrengths = [20]
    for inputStrength in inputStrengths:
        argDict['targetPath'] = os.getcwd()+'/InputOutputBeforeLearning/in_'+str(inputStrength)+'/'
        argDict['gaussianPeak'] = inputStrength
        relNet = threeWay.RelationalNetwork(**argDict)
        relNet.createNetwork()
        relNet.run()
        del relNet
        
    activityDataPath = os.getcwd()+'/InputOutputBeforeLearning/'
    import InputOutputBeforeLearningUtil
    InputOutputBeforeLearningUtil.plotActivity(activityDataPath, inputStrengths)
    


