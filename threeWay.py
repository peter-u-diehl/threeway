'''
Created on 22.07.2016

@author: Peter Udo Diehl
'''
#------------------------------------------------------------------------------ 
# imports and brian options
#------------------------------------------------------------------------------ 
# import matplotlib
# matplotlib.use('Agg')

import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging
import os
import brian as b
from brian import *
b.globalprefs.set_global_preferences( 
                        defaultclock = b.Clock(dt=0.5*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave=True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimisations are turned on 
                        #- if you need IEEE guaranteed results, turn this switch off.
                        useweave_linear_diffeq = False,  # Whether to use weave C++ acceleration for the solution of linear differential 
                        #equations. Note that on some platforms, typically older ones, this is faster and on some platforms, 
                        #typically new ones, this is actually slower.
                        usecodegen = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenreset = True,  # Whether or not to use experimental code generation support on resets. 
                        #Typically slower due to weave overheads, so usually leave this off.
                        usecodegenthreshold = True,  # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp = True,  # Whether or not to use experimental new C STDP.
                        openmp = False,  # Whether or not to use OpenMP pragmas in generated C code. 
                        #If supported on your compiler (gcc 4.2+) it will use multiple CPUs and can run substantially faster.
                        magic_useframes = True,  # Defines whether or not the magic functions should search for objects 
                        #defined only in the calling frame or if they should find all objects defined in any frame. 
                        #This should be set to False if you are using Brian from an interactive shell like IDLE or IPython 
                        #where each command has its own frame, otherwise set it to True.
                       ) 
 
import numpy as np
import matplotlib.cm as cm
import time
import sys  
import getopt
import scipy 


def getArgs():
    #------------------------------------------------------------------------------ 
    # get command line arguments
    #------------------------------------------------------------------------------ 
    argDict = {}
    argv = sys.argv[1:]
    helpMessage = 'test.py --numExamples <integer> --testMode <boolean> --seed <integer> --inputWeight <realNumber>' + \
                    ' --recurrentWeight <realNumber> --targetFiringRate <realNumber> --inputRate <realNumber>' + \
                    ' --nuEEpre <realNumber> --nuEEpost <realNumber> --nuIE <realNumber> --noise <realNumber>' + \
                    ' --inputType <string> --expEEpre <realNumber> --expEEpost <realNumber> --ending <string>' + \
                    ' --gaussianPeakLow <realNumber> --dataPath <path> --targetPath <path> --weightPath <path> --delayPath <path>' + \
                    ' --singleExampleTime <realNumber>'
    try:
        opts, args = getopt.getopt(argv,"h",["numExamples=","testMode=","seed=","inputWeight=","recurrentWeight=",
                                             "targetFiringRate=","inputRate=","nuEEpre=","nuEEpost=","nuIE=","noise=",
                                             "inputType=", "expEEpre=", "expEEpost=", "gaussianPeakLow=", 
                                             "targetPath=", "weightPath=", "delayPath=", "dataPath=", "ending=", "singleExampleTime="])
    except getopt.GetoptError:
        print helpMessage
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helpMessage
            sys.exit()
        elif opt in ("--numExamples"):
            argDict['numExamples'] = int(arg)
        elif opt in ("--testMode"):
            argDict['testMode'] = arg
        elif opt in ("--seed"):
            argDict['seed'] = int(arg)
        elif opt in ("--inputWeight"):
            argDict['inputWeight'] = float(arg)
        elif opt in ("--recurrentWeight"):
            argDict['recurrentWeight'] = float(arg)
        elif opt in ("--targetFiringRate"):
            argDict['targetFiringRate'] = float(arg)
        elif opt in ("--inputRate"):
            argDict['gaussianPeak'] = float(arg)
        elif opt in ("--gaussianPeakLow"):
            argDict['gaussianPeakLow'] = float(arg)
        elif opt in ("--nuEEpre"):
            argDict['nuEEpre'] = float(arg)
        elif opt in ("--nuEEpost"):
            argDict['nuEEpost'] = float(arg)
        elif opt in ("--nuIE"):
            argDict['nuIE'] = float(arg)
        elif opt in ("--noise"):
            argDict['noise'] = float(arg)
        elif opt in ("--inputType"):
            argDict['inputType'] = arg
        elif opt in ("--expEEpre"):
            argDict['expEEpre'] = float(arg)
        elif opt in ("--expEEpost"):
            argDict['expEEpost'] = float(arg)
        elif opt in ("--targetPath"):
            argDict['targetPath'] = arg
        elif opt in ("--dataPath"):
            argDict['dataPath'] = arg
        elif opt in ("--weightPath"):
            argDict['weightPath'] = arg
        elif opt in ("--delayPath"):
            argDict['delayPath'] = arg
        elif opt in ("--ending"):
            argDict['ending'] = arg
        elif opt in ("--singleExampleTime"):
            argDict['singleExampleTime'] = arg
    if not 'targetPath' in argDict:
        argDict['targetPath'] = os.getcwd() + '/' + str(opts) + '/'
    return argDict
        
            
class RelationalNetwork:
    
    def __init__(self, testMode=False, dataPath=None, weightPath=None, delayPath=None, targetPath=None, 
                 numExamples=30000, seed=0, inputWeight=0.125, recurrentWeight=0.01, targetFiringRate=3, 
                 gaussianPeak=20, gaussianPeakLow=None, gaussianDist=None, gaussianSigma=1./6.,
                 nuEEpre=0.0005, nuEEpost=0.0025, nuIE=0.005, 
                 noise=0, ending = '', inputType='add', expEEpre=0.2, expEEpost=0.2,
                 singleExampleTime = 0.25, saveSpikeCountsPerExample=False):
        #------------------------------------------------------------------------------ 
        # set parameters and equations
        #------------------------------------------------------------------------------
        if dataPath==None:
            self.dataPath = os.getcwd()+'/'
        else:
            self.dataPath = dataPath
        if weightPath==None:
            self.weightPath = self.dataPath+'random/'  
        else:
            self.weightPath = weightPath
        if delayPath==None:
            self.delayPath = self.dataPath+'random_delay/'
        else:
            self.delayPath = delayPath

        if not os.path.exists(self.weightPath):
            os.makedirs(self.weightPath)
        if not os.path.exists(self.delayPath):
            os.makedirs(self.delayPath)
        if os.listdir(self.weightPath) == [] or os.listdir(self.delayPath) == []:
            if os.listdir(self.weightPath) == []:
                print '!!!!!!!!!!!!!!!!     random      !!!!!!!!!!!!!!!!' , self.weightPath
                self.weightPath = self.dataPath+'random/'  
            if os.listdir(self.delayPath) == []:
                print '!!!!!!!!!!!!!!!!            random delay           !!!!!!!!!!!!!!!!', self.delayPath
                self.delayPath = self.dataPath+'random_delay/'
            import threeWayConns
            threeWayConns.createRandomWeights(self.dataPath)

        np.random.seed(seed)
        self.ending = ending
        
        self.nE = 1600
        self.nI = self.nE/4
        self.singleExampleTime =  singleExampleTime*b.second #runtime # 
        self.numExamples = numExamples
        self.restingTime = 0.0*b.second
        self.runtime = self.numExamples*(self.singleExampleTime+self.restingTime)
        self.testMode = testMode
        if self.testMode:
            self.normalization_interval = self.numExamples+1
            self.recordSpikes = True
            self.eeSTDPon = False # True # 
            self.plotError = True
            self.restingTime = 0.25*b.second
        else:
            self.normalization_interval = 20
            self.recordSpikes = False
            self.eeSTDPon = True
            self.plotError = True
        self.realTimePlotting = False
        self.saveSpikeCountsPerExample=saveSpikeCountsPerExample
        
        self.v_restE = -65*b.mV 
        self.v_restI = -60*b.mV 
        self.v_resetE = -65.*b.mV
        self.v_resetI = -45.*b.mV
        self.v_threshE = -52.*b.mV
        self.v_threshI = -40.*b.mV
        self.refracE = 5.*b.ms
        self.refracI = 2.*b.ms
        
        self.connStructure = 'sparse' # 'dense' 
        self.weight = {}
        self.delay = {}
        self.weight['ee_input'] = inputWeight
        self.weight['ee'] = recurrentWeight
        self.delay['ee_input'] = (4*b.ms,10*b.ms)
        self.delay['ei_input'] = (0*b.ms,4*b.ms)
        self.delay['ee'] = (2*b.ms,5*b.ms)
        self.delay['ei'] = (0*b.ms,1*b.ms)
        self.delay['ie'] = (0*b.ms,1*b.ms)
        self.delay['ii'] = (0*b.ms,2*b.ms)
        
        self.TCpreEE = 20*b.ms
        self.TCpost1EE = 40*b.ms
        self.TCpost2EE = 40*b.ms
        self.TCpreIE = 20*b.ms
        self.TCpostIE = 20*b.ms
        self.nuEEpre =  nuEEpre      # learning rate
        self.nuEEpost = nuEEpost       # learning rate
        self.nuIE =     nuIE       # learning rate
        self.alphaIE = targetFiringRate*b.Hz*self.TCpostIE*2    # controls the firing rate
        self.wmaxEE = 0.5
        self.wmaxIE = 1000.
        self.expEEpre = expEEpre
        self.expEEpost = expEEpost
        
        self.gaussianPeak = gaussianPeak
        self.gaussianSigma = gaussianSigma
        self.noise = noise
        if gaussianPeakLow != None:
            self.gaussianPeakLow = gaussianPeakLow
        if gaussianDist != None:
            self.gaussianDist = gaussianDist
        
        self.neuronEqsE = '''
                dv/dt = ((v_restE-v) + (I_synE+I_synI) / nS) / (20*ms)  : volt
                I_synE = ge * nS *         -v                           : amp
                I_synI = gi * nS * (-85.*mV-v)                          : amp
                dge/dt = -ge/(5.0*ms)                                   : 1
                dgi/dt = -gi/(10.0*ms)                                  : 1
                '''
        
        self.neuronEqsI = '''
                dv/dt = ((v_restI-v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
                I_synE = ge * nS *         -v                           : amp
                I_synI = gi * nS * (-85.*mV-v)                          : amp
                dge/dt = -ge/(5.0*ms)                                   : 1
                dgi/dt = -gi/(10.0*ms)                                  : 1
                '''
                
        self.eqs_stdpEE = '''
                    post2before                        : 1.0
                    dpre/dt   =   -pre/(TCpreEE)       : 1.0
                    dpost1/dt = -post1/(TCpost1EE)     : 1.0
                    dpost2/dt = -post2/(TCpost2EE)     : 1.0
                    '''
        self.eqs_stdpIE = '''
                    dpre/dt   =  -pre/(TCpreIE)        : 1.0
                    dpost/dt  = -post/(TCpostIE)       : 1.0
                    '''
                    
        if self.expEEpre==0:
            self.eqsSTDPPreEE = 'pre = 1.; w -= nuEEpre * post1'
        else:
            self.eqsSTDPPreEE = 'pre = 1.; w -= nuEEpre * post1 * w**expEEpre'
                    
        self.eqsSTDPPostEE = 'post2before = post2; w += nuEEpost * pre * post2before * (wmaxEE - w)**expEEpost; post1 = 1.; post2 = 1.'
        
        self.eqsSTDPPreIE = 'pre += 1.; w += nuIE * (post-alphaIE)'
        self.eqsSTDPPostIE = 'post += 1.; w += nuIE * pre'
        
        
        self.neuronGroups = {}
        self.inputGroups = {}
        self.connections = {}
        self.STDPMethods = {}
        self.rateMonitors = {}
        self.spikeMonitors = {}
        self.spikeCounters = {}
        self.stateMonitors = {}
        
        self.populationNames = ['A', 'B', 'C', 'H']
        self.inputPopulationNames = ['X', 'Y', 'Z']
        self.inputConnectionNames = ['XA', 'YB', 'ZC']
        self.interPopConnectionNames = ['AH', 'BH', 'CH',
                                  'HA', 'HB', 'HC',
                                  ]
        self.inputConns = ['ee_input', 'ei_input']
        self.recurrentConns = ['ee', 'ei', 'ie', 'ii']
        self.interPopConns = ['ee_input', 'ei_input']


        if inputType == "add":
            import AddInputs
            self.setNewInput = AddInputs.setNewInput
        elif inputType == "equality":
            import EqualityInputs
            self.setNewInput = EqualityInputs.setNewInput
        elif inputType == "negative":
            import NegativeInputs
            self.setNewInput = NegativeInputs.setNewInput
        elif inputType == "doubleSquare":
            import DoubleSquareInputs
            self.setNewInput = DoubleSquareInputs.setNewInput
        elif inputType == "WTA":
            import WTAUtil
            self.setNewInput = WTAUtil.setNewInput
        elif inputType == "CueIntegration":
            import CueIntegrationUtil
            self.setNewInput = CueIntegrationUtil.setNewInput
        elif inputType == "InputOutput":
            import InputOutputUtil
            self.setNewInput = InputOutputUtil.setNewInput
        elif inputType == "SignalRestoration":
            import SignalRestorationUtil
            self.setNewInput = SignalRestorationUtil.setNewInput
        elif inputType == "BerkesFig1":
            import BerkesFig1Util
            self.setNewInput = BerkesFig1Util.setNewInput
            self.saveSpikeCountsPerExample=True
        elif inputType == "BerkesFig2":
            import BerkesFig2Util
            self.setNewInput = BerkesFig2Util.setNewInput
            self.saveSpikeCountsPerExample=True
        elif inputType == "multipleAngles":
            import MultipleAnglesUtil
            self.setNewInput = MultipleAnglesUtil.setNewInput
            self.saveSpikeCountsPerExample=True
        else:
            raise Exception("Inputtype '"+inputType+"' is unknown.'")

        if not targetPath==None:  
            self.dataPath = targetPath
        if not testMode:
            if not os.path.exists(self.dataPath):
                os.makedirs(self.dataPath)
            if not os.path.exists(self.dataPath + 'weights/'):
                os.makedirs(self.dataPath + 'weights/')
            if not os.path.exists(self.dataPath + 'sortedWeights/'):
                os.makedirs(self.dataPath + 'sortedWeights/')
        if not os.path.exists(self.dataPath + 'activity/'):
            os.makedirs(self.dataPath + 'activity/')
            
        self.numPops = len(self.populationNames) + len(self.inputPopulationNames)
        self.numInputPops = len(self.populationNames) + len(self.inputPopulationNames)
        self.numAllPops = self.numPops + self.numInputPops
        self.popValues = np.zeros((self.numExamples, self.numAllPops))
        self.resultMonitor = np.zeros((self.numExamples, self.numAllPops))
    
     
    def createTopoInput(self, nE, popVal, activationFunction=None):
        #------------------------------------------------------------------------------ 
        # create topoligical (in this case Gaussian shaped) input 
        #------------------------------------------------------------------------------    
        if activationFunction == None:
            activationFunction = self.gaussian1D
        centerID = int(popVal*nE)
        topoCoords = {}
        for i in range(nE):
            pos = 1. * float(i)/nE
            topoCoords[i] = (0.5,pos)
        center_coords = topoCoords[centerID]
        dists = np.zeros(nE)
        
        for i in range(nE):
            coords = topoCoords[i]
            deltaX = abs(coords[0]-center_coords[0])
            deltaY = abs(coords[1]-center_coords[1])
            if deltaX > 0.5: deltaX=1.0-deltaX 
            if deltaY > 0.5: deltaY=1.0-deltaY  
            squared_dist = deltaX ** 2  + deltaY  ** 2
            dists[i] = squared_dist
        distsAndIds = zip(dists, range(nE))
        distsAndIds.sort()
        unused_sorted_dists, dist_sorted_ids = zip(*distsAndIds)
        activity = np.zeros(nE)
        for i,idx in enumerate(dist_sorted_ids):
            activity[idx] = activationFunction(float(i)/nE)
        return activity
    
    
    def computePopVector(self, popArray):
        #------------------------------------------------------------------------------ 
        # calculate the circular mean of an array
        #------------------------------------------------------------------------------    
        size = len(popArray)
        complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
        cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
        return cur_pos
    
    
    def gaussian1D(self, x):
        return self.gaussianPeak * (np.exp(-0.5 * (x / self.gaussianSigma)**2))
    
    
    def getMatrixFromFile(self, fileName):
        #------------------------------------------------------------------------------ 
        # read out weight matrix file and return it as 2D numpy-array
        #------------------------------------------------------------------------------    
        if 'delay' in fileName:
            offset = 0
        else:
            offset = len(self.ending)
        if fileName[-3-4-offset]=='e':
            nSrc = self.nE
        else:
            nSrc = self.nI
        if fileName[-1-4-offset]=='e':
            nTgt = self.nE
        else:
            nTgt = self.nI
        readout = np.load(fileName)
        value_arr = np.zeros((nSrc, nTgt))
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
        return value_arr, readout
    
    
    def saveConnections(self, ending=''):
        #------------------------------------------------------------------------------ 
        # save all weight matricies to files
        #------------------------------------------------------------------------------    
        print 'save connections'
        for connName in self.connections:
            connMatrix = self.connections[connName][:]
            connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
            np.save(self.dataPath + 'weights/' + connName + ending, connListSparse)
    
    
    def saveDelays(self, ending=''):
        #------------------------------------------------------------------------------ 
        # save all weight matricies to files
        #------------------------------------------------------------------------------    
        print 'save delays'
        for connName in self.connections:
            if connName == 'ZeCi':
                if connName[1]=='e':
                    nSrc = self.nE
                else:
                    nSrc = self.nI
                if connName[3]=='e':
                    nTgt = self.nE
                else:
                    nTgt = self.nI
                connListSparse = ([(i,j,self.connections[connName].delay[i,j]) for i in xrange(nSrc) for j in xrange(nTgt) if self.connections[connName].delay[i,j]])
                np.savetxt(self.dataPath + 'delays/' + connName + ending, connListSparse)
    
    
    def normalizeWeights(self):
        #------------------------------------------------------------------------------ 
        # normalize all excitatory to excitatory weight matricies row and column wise 
        #------------------------------------------------------------------------------    
        print 'normalize weights'
        for connName in self.connections:
            if connName[1] == 'e' and connName[3] == 'e':
                if connName[0] == connName[2]:   # ==> recurrent connection
                    factor = self.weight['ee']
                else:   # ==> input connection
                    factor = self.weight['ee_input']
                    if connName[2] == 'H':
                        factor *= 0.66
                        
                connection = self.connections[connName][:]
                
#                 w_pre = np.zeros((self.nE, self.nE))
                w_post = np.zeros((self.nE, self.nE))
                for i in xrange(self.nE):#
                    rowi = connection.rowdata[i]
                    rowMean = np.mean(rowi)
#                     w_pre[i, connection.rowj[i]] = rowi
                    connection.rowdata[i] *= factor/rowMean
                    w_post[i, connection.rowj[i]] = connection.rowdata[i]
                
                colMeans = np.sum(w_post, axis = 0)
                colFactors = factor/colMeans
                colDataEntries = [len(connection.coldataindices[j]) for j in xrange(self.nE)]
                
#                 print 'connection[:,0]', np.average(connection[:,0])
#                 print 'target: ', factor, 'connection[0,:]', np.average(self.connections[connName][:][0,:])
                for j in xrange(self.nE):#
                    connection[:,j] *= colFactors[j]*colDataEntries[j]
    
    
    def createNetwork(self):   
        #------------------------------------------------------------------------------ 
        # create network
        #------------------------------------------------------------------------------                 
        self.createNetworkPopulations()
        self.createInputPopulations()
        self.createInputConnections()
        self.createRecurrentConnections()
        self.createInterPopConnections()
            
            
    def createNetworkPopulations(self):
        #------------------------------------------------------------------------------ 
        # create network populations
        #------------------------------------------------------------------------------ 
        self.net = b.Network()
        v_restE = self.v_restE
        v_restI = self.v_restI
        self.neuronGroups['e'] = b.NeuronGroup(self.nE*len(self.populationNames), self.neuronEqsE, 
                                               threshold= self.v_threshE, refractory= self.refracE, 
                                               reset= self.v_resetE, compile = True, freeze = True)
        self.neuronGroups['i'] = b.NeuronGroup(self.nI*len(self.populationNames), self.neuronEqsI, 
                                               threshold= self.v_threshI, refractory= self.refracI, 
                                               reset= self.v_resetI, compile = True, freeze = True)
        self.net.add(self.neuronGroups['e'])
        self.net.add(self.neuronGroups['i'])
        
        for name in self.populationNames:
            print 'create neuron group', name            
            self.neuronGroups[name+'e'] = self.neuronGroups['e'].subgroup(self.nE)
            self.neuronGroups[name+'i'] = self.neuronGroups['i'].subgroup(self.nI)
            self.neuronGroups[name+'e'].v = self.v_restE
            self.neuronGroups[name+'i'].v = self.v_restI
            
            print 'create monitors for', name
            self.rateMonitors[name+'e'] = b.PopulationRateMonitor(self.neuronGroups[name+'e'], 
                                                                  bin = (self.singleExampleTime+self.restingTime)/b.second)
            self.rateMonitors[name+'i'] = b.PopulationRateMonitor(self.neuronGroups[name+'i'], 
                                                                  bin = (self.singleExampleTime+self.restingTime)/b.second)
            self.spikeCounters[name+'e'] = b.SpikeCounter(self.neuronGroups[name+'e'])
            self.spikeCounters[name+'i'] = b.SpikeCounter(self.neuronGroups[name+'i'])
            self.net.add(self.rateMonitors[name+'e'])
            self.net.add(self.rateMonitors[name+'i'])
            self.net.add(self.spikeCounters[name+'e'])
            self.net.add(self.spikeCounters[name+'i'])
            
            
            if self.recordSpikes:
#                 if name == 'C' or name == 'H':  # name == 'A' or name == 'B' or 
                self.spikeMonitors[name+'e'] = b.SpikeMonitor(self.neuronGroups[name+'e'])
                self.spikeMonitors[name+'i'] = b.SpikeMonitor(self.neuronGroups[name+'i'])
                self.net.add(self.spikeMonitors[name+'e'])
                self.net.add(self.spikeMonitors[name+'i'])
        


    def createInputPopulations(self):
        #------------------------------------------------------------------------------ 
        # create input populations
        #------------------------------------------------------------------------------ 
        for i,name in enumerate(self.inputPopulationNames):
            print 'create input group', name
            self.inputGroups[name+'e'] = b.PoissonGroup(self.nE, np.zeros(self.nE))
            self.rateMonitors[name+'e'] = b.PopulationRateMonitor(self.inputGroups[name+'e'], bin = (self.singleExampleTime+self.restingTime)/b.second)
            self.spikeCounters[name+'e'] = b.SpikeCounter(self.inputGroups[name+'e'])
            self.net.add(self.inputGroups[name+'e'])
            self.net.add(self.rateMonitors[name+'e'])
            self.net.add(self.spikeCounters[name+'e'])



    def createInputConnections(self):
        #------------------------------------------------------------------------------ 
        # create connections from input populations to network populations
        #------------------------------------------------------------------------------ 
        for name in self.inputConnectionNames:
            print 'create connections between', name[0], 'and', name[1]
            for connType in self.inputConns:                
                connName = name[0] + connType[0] + name[1] + connType[1]
                weightMatrix, unused_weightList = self.getMatrixFromFile(self.weightPath+connName+self.ending+'.npy')
                weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
                self.connections[connName] = b.Connection(self.inputGroups[connName[0:2]], self.neuronGroups[connName[2:4]], structure= self.connStructure, 
                                                            state = 'g'+connType[0], delay=True, max_delay=self.delay[connType][1])
                self.connections[connName].connect(self.inputGroups[connName[0:2]], self.neuronGroups[connName[2:4]], weightMatrix)
                delayMatrix, unused_delayList = self.getMatrixFromFile(self.delayPath+connName+'.npy')
                delayMatrix = scipy.sparse.lil_matrix(delayMatrix)
                nonZeroDelays = np.nonzero(delayMatrix)
                self.connections[connName].delay[nonZeroDelays] = delayMatrix[nonZeroDelays]*b.ms
                self.net.add(self.connections[connName])
            
            if self.eeSTDPon:
                TCpreEE = self.TCpreEE
                TCpost1EE = self.TCpost1EE
                TCpost2EE = self.TCpost2EE
                nuEEpre = self.nuEEpre
                nuEEpost = self.nuEEpost
                wmaxEE = self.wmaxEE
                expEEpre = self.expEEpre
                expEEpost = self.expEEpost
                self.STDPMethods[name[0]+'e'+name[1]+'e'] =  b.STDP(self.connections[name[0]+'e'+name[1]+'e'], eqs=self.eqs_stdpEE, pre=self.eqsSTDPPreEE, 
                                                                    post=self.eqsSTDPPostEE, wmin=0., wmax= self.wmaxEE)
                self.net.add(self.STDPMethods[name[0]+'e'+name[1]+'e'])



    def createRecurrentConnections(self):
        #------------------------------------------------------------------------------ 
        # create recurrent connections
        #------------------------------------------------------------------------------ 
        for name in self.populationNames:
            print 'create recurrent connections for population', name
            for connType in self.recurrentConns:
                connName = name+connType[0]+name+connType[1]
                weightMatrix, unused_weightList = self.getMatrixFromFile(self.weightPath+connName+self.ending+'.npy')
                weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
                self.connections[connName] = b.Connection(self.neuronGroups[connName[0:2]], self.neuronGroups[connName[2:4]], structure= self.connStructure, 
                                                            state = 'g'+connType[0], delay=True, max_delay=self.delay[connType][1])
                self.connections[connName].connect(self.neuronGroups[connName[0:2]], self.neuronGroups[connName[2:4]], weightMatrix)
                delayMatrix, unused_delayList = self.getMatrixFromFile(self.delayPath+connName+'.npy')
                delayMatrix = scipy.sparse.lil_matrix(delayMatrix)
                nonZeroDelays = np.nonzero(delayMatrix)
                self.connections[connName].delay[nonZeroDelays] = delayMatrix[nonZeroDelays]*b.ms
                self.net.add(self.connections[connName])
                    
            
            print 'create STDP for', name
            if self.eeSTDPon:
                TCpreEE = self.TCpreEE
                TCpost1EE = self.TCpost1EE
                TCpost2EE = self.TCpost2EE
                nuEEpre = self.nuEEpre
                nuEEpost = self.nuEEpost
                wmaxEE = self.wmaxEE
                expEEpre = self.expEEpre
                expEEpost = self.expEEpost
                self.STDPMethods[name+'e'+name+'e'] = b.STDP(self.connections[name+'e'+name+'e'], eqs=self.eqs_stdpEE, pre=self.eqsSTDPPreEE, 
                                                             post=self.eqsSTDPPostEE, wmin=0., wmax=self.wmaxEE)
                self.net.add(self.STDPMethods[name+'e'+name+'e'])
            if not self.testMode:
                TCpreIE = self.TCpreIE
                TCpostIE = self.TCpostIE
                nuIE = self.nuIE
                alphaIE = self.alphaIE
                wmaxIE = self.wmaxIE
                self.STDPMethods[name+'i'+name+'e'] = b.STDP(self.connections[name+'i'+name+'e'], eqs=self.eqs_stdpIE, pre=self.eqsSTDPPreIE, 
                                                             post=self.eqsSTDPPostIE, wmin=0., wmax=self.wmaxIE)
                self.net.add(self.STDPMethods[name+'i'+name+'e'])



    def createInterPopConnections(self):
        #------------------------------------------------------------------------------ 
        # create connections between populations
        #------------------------------------------------------------------------------ 
        for name in self.interPopConnectionNames:
            print 'create connections between', name[0], 'and', name[1]
            for connType in self.interPopConns:
                connName = name[0] + connType[0] + name[1] + connType[1]
                weightMatrix, unused_weightList = self.getMatrixFromFile(self.weightPath+connName+self.ending+'.npy')
                weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
                self.connections[connName] = b.Connection(self.neuronGroups[connName[0:2]], self.neuronGroups[connName[2:4]], structure= self.connStructure, 
                                                            state = 'g'+connType[0], delay=True, max_delay=self.delay[connType][1])
                self.connections[connName].connect(self.neuronGroups[connName[0:2]], self.neuronGroups[connName[2:4]], weightMatrix)
                delayMatrix, unused_delayList = self.getMatrixFromFile(self.delayPath+connName+'.npy')
                delayMatrix = scipy.sparse.lil_matrix(delayMatrix)
                nonZeroDelays = np.nonzero(delayMatrix)
                self.connections[connName].delay[nonZeroDelays] = delayMatrix[nonZeroDelays]*b.ms
                self.net.add(self.connections[connName])
                
            if self.eeSTDPon:
                TCpreEE = self.TCpreEE
                TCpost1EE = self.TCpost1EE
                TCpost2EE = self.TCpost2EE
                nuEEpre = self.nuEEpre
                nuEEpost = self.nuEEpost
                wmaxEE = self.wmaxEE
                expEEpre = self.expEEpre
                expEEpost = self.expEEpost
                self.STDPMethods[name[0]+'e'+name[1]+'e'] = b.STDP(self.connections[name[0]+'e'+name[1]+'e'], eqs=self.eqs_stdpEE, pre=self.eqsSTDPPreEE, 
                                                       post=self.eqsSTDPPostEE, wmin=0., wmax=self.wmaxEE)
                self.net.add(self.STDPMethods[name[0]+'e'+name[1]+'e'])
    


    def run(self):
        #------------------------------------------------------------------------------ 
        # run the simulation and set inputs
        #------------------------------------------------------------------------------ 
        previousSpikeCount = np.zeros((self.nE, self.numAllPops))
        currentSpikeCount = np.zeros((self.nE, self.numAllPops))
        if self.saveSpikeCountsPerExample:
            self.spikeCounts=np.zeros((self.numExamples, self.nE, self.numAllPops))
        
        if self.realTimePlotting and self.recordSpikes:
            b.ion()
            fig = b.figure(1)
            b.raster_plot(self.spikeMonitors['Ae'])
    
        if self.ending=='':
            initJ = 0
        else:
            initJ = int(self.ending)
        for j in xrange(int(self.numExamples)):            
            if self.restingTime or j==0:
                for i,name in enumerate(self.inputPopulationNames):
                    rates = np.ones(self.nE)  * 0
                    self.inputGroups[name+'e'].rate = rates
                self.net.run(self.restingTime)
                
            if j%self.normalization_interval == 0:
                self.normalizeWeights()
                
            print 'set new rates of the inputs'
            self.setNewInput(self,j)
                    
            print 'run number:', j+1, 'of', int(self.numExamples)
            self.net.run(self.singleExampleTime)
            for i,name in enumerate(self.populationNames):
                name += 'e'
                currentSpikeCount[:,i] = np.asarray(self.spikeCounters[name].count[:]) - previousSpikeCount[:,i]
                previousSpikeCount[:,i] = np.copy(self.spikeCounters[name].count[:])
                self.resultMonitor[j,i] = self.computePopVector(currentSpikeCount[:,i])
                print name, 'pop. activity: ', self.resultMonitor[j,i], ', spikecount:', sum(currentSpikeCount[:,i])
                if self.saveSpikeCountsPerExample:
                    self.spikeCounts[j,:,i] = currentSpikeCount[:,i]
                    
            for i,name in enumerate(self.inputPopulationNames):
                print name, 'pop. activity: ', (self.popValues[j,i])
                name += 'e'
                currentSpikeCount[:,i+self.numPops] = np.asarray(self.spikeCounters[name].count[:]) - previousSpikeCount[:,i+self.numPops]
                previousSpikeCount[:,i+self.numPops] = np.copy(self.spikeCounters[name].count[:])
                self.resultMonitor[j,i+self.numPops] = self.computePopVector(currentSpikeCount[:,i+self.numPops])
                if self.saveSpikeCountsPerExample:
                    self.spikeCounts[j,:,i+self.numPops] = currentSpikeCount[:,i+self.numPops]
                
                    
            if not self.testMode:
                if self.numExamples <= 1000:
                    if (j+1)%100==0 and not j==0:
                        self.saveConnections(str(j+initJ+1))
                else:
                    if (j+1)%5000==0 and not j==0:
                        self.saveConnections(str(j+initJ+1))
                        
            if self.realTimePlotting and self.recordSpikes:
                b.raster_plot(self.spikeMonitors['Ae'], showlast=1000*b.ms)
                fig.canvas.draw()
            
        
        #------------------------------------------------------------------------------ 
        # save results
        #------------------------------------------------------------------------------ 
        print 'save results'
        
        if self.testMode:
            np.savetxt(self.dataPath + 'activity/resultPopVecs' + str(self.numExamples) + '.txt', self.resultMonitor)
            np.savetxt(self.dataPath + 'activity/popVecs' + str(self.numExamples) + '.txt', self.popValues)
            for name in self.spikeCounters:
                np.savetxt(self.dataPath + 'activity/spikeCount' + name + '.txt', 
                           self.spikeCounters[name].count[:]/(self.singleExampleTime*int(self.numExamples)))
            if self.saveSpikeCountsPerExample:
                np.save(self.dataPath + 'activity/spikeCountPerExample', self.spikeCounts)
        else:
            self.saveConnections(str(self.numExamples+initJ))
            self.normalizeWeights()
            self.saveConnections()



    def sortWeights(self):
        #------------------------------------------------------------------------------ 
        # sort weights
        #------------------------------------------------------------------------------ 
        import threeWaySortWeights
        ending = ''
        srcWeightPath = self.dataPath + 'weights/'
        tgtWeightPath = self.dataPath + 'sortedWeights/'
        threeWaySortWeights.saveSorted(self.nE, self.nI, srcWeightPath, tgtWeightPath, 
                   ending, self.inputConnectionNames, self.interPopConnectionNames)



if __name__ == "__main__":
    #------------------------------------------------------------------------------ 
    # get arguments
    #------------------------------------------------------------------------------ 
    argDict = getArgs()



    #------------------------------------------------------------------------------ 
    # train network
    #------------------------------------------------------------------------------ 
    if 'testMode' not in argDict or not argDict['testMode'] == 'True':
        print '____________ training ____________'
        relNet = RelationalNetwork(**argDict)
        relNet.createNetwork()
        relNet.run()
        relNet.sortWeights()
        dataPath = relNet.dataPath
        #------------------------------------------------------------------------------ 
        # visualize results
        #------------------------------------------------------------------------------ 
        import WeightReadout
        import HiddenLayerVisualization
        WeightReadout.plotWeights(dataPath+'sortedWeights/')
        HiddenLayerVisualization.plotWeights(dataPath+'sortedWeights/')
        del relNet
    else:
        dataPath = './'


    #------------------------------------------------------------------------------ 
    # test network
    #------------------------------------------------------------------------------ 
    numTestingExamples = 1000
    argDict['numExamples'] = numTestingExamples 
    if 'weightPath' not in argDict:
        argDict['weightPath'] = dataPath+'sortedWeights/'
    testingNoise = 0
    argDict['noise'] = testingNoise
    argDict['testMode'] = True
    print '____________ testing ____________'
    relNet = RelationalNetwork(**argDict)
    relNet.createNetwork()
    relNet.run()



    #------------------------------------------------------------------------------ 
    # visualize results
    #------------------------------------------------------------------------------ 
    activityDataPath = dataPath + 'activity/'
    if 'inputType' in argDict:
        if argDict['inputType'] == "add":
            import AddInputs
            AddInputs.evaluate(activityDataPath, ending=str(numTestingExamples))
        elif argDict['inputType'] == "equality":
            import EqualityInputs
            EqualityInputs.evaluate(activityDataPath, ending=str(numTestingExamples))
        elif argDict['inputType'] == "doubleSquare":
            import DoubleSquareInputs
            DoubleSquareInputs.evaluate(activityDataPath, ending=str(numTestingExamples))
        elif argDict['inputType'] == "negative":
            import NegativeInputs
            NegativeInputs.evaluate(activityDataPath, ending=str(numTestingExamples))
    else:
        import EqualityInputs
        EqualityInputs.evaluate(activityDataPath, ending=str(numTestingExamples))

    import generateFigures
    print argDict['weightPath']
    generateFigures.generateAllFigs(argDict)

