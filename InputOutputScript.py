import os
import threeWay
import numpy as np

def simulateNetwork(inputStrengths):
    argDict = threeWay.getArgs()
    print argDict

    numTestingExamples = 1
    testingNoise = 0
    argDict['numExamples'] = numTestingExamples
    argDict['testMode'] = True
    argDict['weightPath'] = os.getcwd()+'/sortedWeights/'
    argDict['noise'] = testingNoise
    argDict['inputType'] = 'InputOutput'
    argDict['inputWeight'] = 0.12
    argDict['recurrentWeight'] = 0.001
    argDict['singleExampleTime'] = 5.0
    
    for inputStrength in inputStrengths:
        argDict['targetPath'] = os.getcwd()+'/InputOutput/in_'+str(inputStrength)+'/'
        argDict['gaussianPeak'] = inputStrength
        relNet = threeWay.RelationalNetwork(**argDict)
        relNet.createNetwork()
        relNet.run()
        del relNet

if __name__ == "__main__":
    inputStrengths = np.logspace(-0.5, 2.3, 20, base=10)
#     simulateNetwork(inputStrengths)
        
    activityDataPath = os.getcwd()+'/InputOutput/'
    import InputOutputUtil
    InputOutputUtil.plotActivity(activityDataPath, inputStrengths)
    


