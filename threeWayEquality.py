'''
Created on 05.11.2012

@author: peter
'''
#------------------------------------------------------------------------------ 
# imports and brian options
#------------------------------------------------------------------------------ 

import matplotlib
matplotlib.use('Agg')

import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging
import os
import brian as b
from brian import *
# from brian.globalprefs import *
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
import matplotlib
import matplotlib.cm as cm
import time
import sys  
import getopt
import scipy 
import brian.experimental.realtime_monitor as rltmMon

# import brian.experimental.cuda.gpucodegen as gpu


            
class RelationalNetwork:
    
    def __init__(self):
        #------------------------------------------------------------------------------ 
        # set parameters and equations
        #------------------------------------------------------------------------------
        self.dataPath = os.getcwd() + '/'
        self.weightPath = self.dataPath +   'random/'#     'sortedWeights/'#       'weights/'#       'random/'#   
        np.random.seed(0)
        
        self.nE = 1600
        self.nI = self.nE/4
        self.singleExampleTime =  0.25*b.second #runtime # 
        self.numExamples = 1 * 1
        self.restingTime = 0.0*b.second
        self.runtime = self.numExamples*(self.singleExampleTime+self.restingTime)
        # defaultclock.dt = 0.2*ms
        self.normalization_interval = 20
        self.testMode = False
        if self.testMode:
            self.recordSpikes = True
            self.eeSTDPon = False # True # 
            self.plotError = True
        else:
            self.recordSpikes = False
            self.eeSTDPon = True
            self.plotError = True
        
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
        
        self.weight['ee_input'] = 0.12 # 0.10
        self.weight['ee'] = 0.01
        
        self.delay['ee_input'] = (0*b.ms,10*b.ms)
        self.delay['ei_input'] = (0*b.ms,5*b.ms)
        self.delay['ee'] = (0*b.ms,5*b.ms)
        self.delay['ei'] = (0*b.ms,2*b.ms)
        self.delay['ie'] = (0*b.ms,1*b.ms)
        self.delay['ii'] = (0*b.ms,2*b.ms)
        
        self.TCpreEE = 20*b.ms
        self.TCpost1EE = 40*b.ms
        self.TCpost2EE = 40*b.ms
        self.TCpreIE = 20*b.ms
        self.TCpostIE = 20*b.ms
        self.nuEEpre =  0.0005      # learning rate
        self.nuEEpost = 0.0025       # learning rate
        self.nuIE =     0.005       # learning rate
        self.nuII =     0.005       # learning rate
        self.alphaIE = 2*b.Hz*self.TCpostIE*2    # controls the firing rate
        self.wmaxEE = 0.5
        self.wmaxIE = 1000.
        self.wmaxII = 1.0
        self.expEEpre = 0.2
        self.expEEpost = self.expEEpre
        
        
        self.gaussian_peak = 20
        self.gaussian_sigma = 1./6.
        self.noise = 5
        
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


        #------------------------------------------------------------------------------ 
        # get command line arguments
        #------------------------------------------------------------------------------ 
        argv = sys.argv[1:]
        try:
            opts, args = getopt.getopt(argv,"h",["numExamples=","testMode=","seed=","inputWeight=","recurrentWeight=",
                                                 "targetFiringRate=","inputRate=","nuEEpre=","nuEEpost=","nuIE=","noise="])
        except getopt.GetoptError:
            print 'test.py --numExamples <integer> --testMode <boolean> --seed <integer> --inputWeight <realNumber>' + \
                        ' --recurrentWeight <realNumber> --targetFiringRate <realNumber> --inputRate <realNumber>' + \
                        ' --nuEEpre <realNumber> --nuEEpost <realNumber> --nuIE <realNumber> --noise <realNumber>'
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print 'test.py --numExamples <integer> --testMode <boolean> --seed <integer> --inputWeight <realNumber>' + \
                            ' --recurrentWeight <realNumber> --targetFiringRate <realNumber> --inputRate <realNumber>' + \
                            ' --nuEEpre <realNumber> --nuEEpost <realNumber> --nuIE <realNumber> --noise <realNumber>'
                sys.exit()
            elif opt in ("--numExamples"):
                self.numExamples = int32(arg)
            elif opt in ("--testMode"):
                self.testMode = arg
            elif opt in ("--seed"):
                np.random.seed(arg)
            elif opt in ("--inputWeight"):
                self.weight['ee_input'] = arg
            elif opt in ("--recurrentWeight"):
                self.weight['ee'] = arg
            elif opt in ("--targetFiringRate"):
                self.alphaIE = 2*b.Hz*self.TCpostIE*arg
            elif opt in ("--inputRate"):
                self.gaussian_peak = arg
            elif opt in ("--nuEEpre"):
                self.nuEEpre = arg
            elif opt in ("--nuEEpost"):
                self.nuEEpost = arg
            elif opt in ("--nuIE"):
                self.nuIE = arg
            elif opt in ("--noise"):
                self.noise = arg
        
        self.dataPath += str(opts) + '/' + sys.argv[0] + '/'
        if not os.path.exists(self.dataPath):
            os.makedirs(self.dataPath)
        if not os.path.exists(self.dataPath + 'weights/'):
            os.makedirs(self.dataPath + 'weights/')
        if not os.path.exists(self.dataPath + 'sortedWeights/'):
            os.makedirs(self.dataPath + 'sortedWeights/')
        if not os.path.exists(self.dataPath + 'activity/'):
            os.makedirs(self.dataPath + 'activity/')
            
            
    
     
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
            if deltaX > 0.5: deltaX=1.0-deltaX  # silent assumption: topo is defined in unit square (and fills it)
            if deltaY > 0.5: deltaY=1.0-deltaY  # silent assumption: topo is defined in unit square (and fills it)
            squared_dist = deltaX ** 2  + deltaY  ** 2
            dists[i] = squared_dist
        distsAndIds = zip(dists, range(nE))
        distsAndIds.sort()
        unused_sorted_dists, dist_sorted_ids = zip(*distsAndIds)
        activity = np.zeros(nE)
        for i,idx in enumerate(dist_sorted_ids):
            activity[idx] = activationFunction(float(i)/nE)
    #        print "Integral over input activity: %f"%np.sum(activity)
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
        return self.gaussian_peak * (np.exp(-0.5 * (x / self.gaussian_sigma)**2))
    
    def getMatrixFromFile(self, fileName):
        #------------------------------------------------------------------------------ 
        # read out weight matrix file and return it as 2D numpy-array
        #------------------------------------------------------------------------------    
        if fileName[-3-4]=='e':
            nSrc = self.nE
        else:
            nSrc = self.nI
        if fileName[-1-4]=='e':
            nTgt = self.nE
        else:
            nTgt = self.nI
        readout = np.load(fileName)
        value_arr = np.zeros((nSrc, nTgt))
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
        return value_arr
    
    def saveConnections(self, ending=''):
        #------------------------------------------------------------------------------ 
        # save all weight matricies to files
        #------------------------------------------------------------------------------    
        print 'save connections'
        for connName in self.connections:
            connMatrix = self.connections[connName][:]
            connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
            np.save(self.dataPath + 'weights/' + connName + ending, connListSparse)
    
    
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
                
                w_pre = np.zeros((self.nE, self.nE))
                w_post = np.zeros((self.nE, self.nE))
                for i in xrange(self.nE):#
                    rowi = connection.rowdata[i]
                    rowMean = np.mean(rowi)
                    w_pre[i, connection.rowj[i]] = rowi
                    connection.rowdata[i] *= factor/rowMean
                    w_post[i, connection.rowj[i]] = connection.rowdata[i]
                
                colMeans = np.sum(w_post, axis = 0)
                colFactors = factor/colMeans
                colDataEntries = [len(connection.coldataindices[j]) for j in xrange(self.nE)]
                
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
        #     neuronGroups[name+'e'] = NeuronGroup(nE, neuronEqsE, threshold= v_threshE, refractory= refracE, reset= v_resetE, 
        #                                          compile = True, freeze = True)
        #     neuronGroups[name+'i'] = NeuronGroup(nI, neuronEqsI, threshold= v_threshI, refractory= refracI, reset= v_resetI, 
        #                                          compile = True, freeze = True)
            
            self.neuronGroups[name+'e'] = self.neuronGroups['e'].subgroup(self.nE)
            self.neuronGroups[name+'i'] = self.neuronGroups['i'].subgroup(self.nI)
#             self.net.add(self.neuronGroups[name+'e'])
#             self.net.add(self.neuronGroups[name+'i'])
            
            self.neuronGroups[name+'e'].v = self.v_restE
            self.neuronGroups[name+'i'].v = self.v_restI
            
            print 'create monitors for', name
            self.rateMonitors[name+'e'] = b.PopulationRateMonitor(self.neuronGroups[name+'e'], 
                                                                  bin = (self.singleExampleTime+self.restingTime)/b.second)
            self.rateMonitors[name+'i'] = b.PopulationRateMonitor(self.neuronGroups[name+'i'], 
                                                                  bin = (self.singleExampleTime+self.restingTime)/b.second)
            self.spikeCounters[name+'e'] = b.SpikeCounter(self.neuronGroups[name+'e'])
            self.net.add(self.rateMonitors[name+'e'])
            self.net.add(self.rateMonitors[name+'i'])
            self.net.add(self.spikeCounters[name+'e'])
            
            if self.recordSpikes:
                if name == 'C' or name == 'H':  # name == 'A' or name == 'B' or 
                    self.spikeMonitors[name+'e'] = b.SpikeMonitor(self.neuronGroups[name+'e'])
                    self.spikeMonitors[name+'i'] = b.SpikeMonitor(self.neuronGroups[name+'i'])
                    self.net.add(self.spikeMonitors[name+'e'])
                    self.net.add(self.spikeMonitors[name+'i'])
        
        #     if name == 'A' or name == 'H':
        #         stateMonitors[name+'e'] = MultiStateMonitor(neuronGroups[name+'e'], ['v', 'ge'], record=[0])
        #         stateMonitors[name+'i'] = MultiStateMonitor(neuronGroups[name+'i'], ['v', 'ge'], record=[0])


    def createInputPopulations(self):
        #------------------------------------------------------------------------------ 
        # create input populations
        #------------------------------------------------------------------------------ 
        for i,name in enumerate(self.inputPopulationNames):
            print 'create input group', name
            self.inputGroups[name+'e'] = PoissonGroup(self.nE, np.zeros(self.nE))
            self.rateMonitors[name+'e'] = b.PopulationRateMonitor(self.inputGroups[name+'e'], bin = (self.singleExampleTime+self.restingTime)/b.second)
#             spikeMonitors[name+'e'] = SpikeMonitor(inputGroups[name+'e'])
            self.net.add(self.inputGroups[name+'e'])
            self.net.add(self.rateMonitors[name+'e'])


    def createInputConnections(self):
        #------------------------------------------------------------------------------ 
        # create connections from input populations to network populations
        #------------------------------------------------------------------------------ 
        for name in self.inputConnectionNames:
            print 'create connections between', name[0], 'and', name[1]
            for connType in self.inputConns:
                connName = name[0] + connType[0] + name[1] + connType[1]
                weightMatrix = self.getMatrixFromFile(self.weightPath+connName+'.npy')
                weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
                self.connections[connName] = Connection(self.inputGroups[connName[0:2]], self.neuronGroups[connName[2:4]], structure= self.connStructure, 
                                                            state = 'g'+connType[0], delay=True, max_delay=self.delay[connType][1])
                self.connections[connName].connect(self.inputGroups[connName[0:2]], self.neuronGroups[connName[2:4]], weightMatrix, delay=self.delay[connType])
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
                weightMatrix = self.getMatrixFromFile(self.weightPath +connName+'.npy')
        #         delayMatrix = np.load(dataPath +'threeWayConnectionMatrix_d'+connName+'.npy')
        #         print weightMatrix.shape, delayMatrix.shape
                weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
        #         delayMatrix = scipy.sparse.lil_matrix(delayMatrix)
                self.connections[connName] = Connection(self.neuronGroups[connName[0:2]], self.neuronGroups[connName[2:4]], structure= self.connStructure, 
                                                            state = 'g'+connType[0], delay=True, max_delay=self.delay[connType][1])#, delay=delay[connType])
                self.connections[connName].connect(self.neuronGroups[connName[0:2]], self.neuronGroups[connName[2:4]], weightMatrix, delay=self.delay[connType])
        #         connections[connName].connect_from_sparse(weightMatrix)#, delay = delayMatrix)
        #         for i in xrange(len(nonZeroDelays[0])):
        #             connections[connName].delay[nonZeroDelays[0][i],nonZeroDelays[1][i]] = delayMatrix[nonZeroDelays[0][i],nonZeroDelays[1][i]]
        #         nonZeroDelays = np.nonzero(delayMatrix)
        #         connections[connName].delay[nonZeroDelays] = delayMatrix[nonZeroDelays]
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
                weightMatrix = self.getMatrixFromFile(self.weightPath+connName+'.npy')
                weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
                self.connections[connName] = Connection(self.neuronGroups[connName[0:2]], self.neuronGroups[connName[2:4]], structure= self.connStructure, 
                                                            state = 'g'+connType[0], delay=True, max_delay=self.delay[connType][1])
                self.connections[connName].connect(self.neuronGroups[connName[0:2]], self.neuronGroups[connName[2:4]], weightMatrix, delay=self.delay[connType])
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
        previousSpikeCountB = np.zeros(self.nE)
        previousSpikeCountC = np.zeros(self.nE)
        self.resultMonitor = np.zeros((self.numExamples,len(self.populationNames)))
        start = time.time()
        
        if self.recordSpikes:
            b.figure()
            b.ion()
            b.subplot(211)
            b.raster_plot(self.spikeMonitors['He'], refresh=1000*b.ms, showlast=1000*b.ms)
            b.subplot(212)
            b.raster_plot(self.spikeMonitors['Hi'], refresh=1000*b.ms, showlast=1000*b.ms)
        
#         realTimeMonitor = None
#         realTimeMonitor = rltmMon.RealtimeConnectionMonitor(self.connections['HeAe'], cmap=cm.get_cmap('gist_rainbow'), 
#                                                             wmin=0, wmax=self.wmaxEE, clock=Clock(1000*b.ms))
        
        
        for j in xrange(int(self.numExamples)):            
            if self.restingTime or j==0:
                for i,name in enumerate(self.inputPopulationNames):
                    rates = np.ones(self.nE)  * 0
                    self.inputGroups[name+'e'].rate = rates
                self.net.run(self.restingTime)#, report='text')
                
            if j%self.normalization_interval == 0:
                self.normalizeWeights()
                
            print 'set new rates of the inputs'
            self.popValues = [0]*len(self.inputPopulationNames)
            for i,name in enumerate(self.inputPopulationNames):
                if name == 'X':
                    self.popValues[i] = np.random.rand();
                    rates = self.createTopoInput(self.nE, self.popValues[i])
                    self.resultMonitor[j,0] = self.popValues[i]
                else:
                    if self.testMode:
                        rates = np.ones(self.nE)  * 0
                    elif name == 'Y':
                        self.popValues[i] = (self.popValues[0]*2) % 1.
                        rates = self.createTopoInput(self.nE, self.popValues[i])
                    elif name == 'Z':
                        self.popValues[i] = (self.popValues[0]**2) % 1.
                        rates = self.createTopoInput(self.nE, self.popValues[i])
                if self.testMode:
                    rates += noise
                self.inputGroups[name+'e'].rate = rates 
                    
                    
            print 'run number:', j+1, 'of', int(self.numExamples)
            self.net.run(self.singleExampleTime)#, report='text')
            currentSpikeCountB = np.asarray(self.spikeCounters['Be'].count[:]) - previousSpikeCountB
            currentSpikeCountC = np.asarray(self.spikeCounters['Ce'].count[:]) - previousSpikeCountC
        #     print currentSpikeCount,  np.asarray(spikeCounters['Ce'].count[:]), previousSpikeCount
            previousSpikeCountB = np.copy(self.spikeCounters['Be'].count[:])
            previousSpikeCountC = np.copy(self.spikeCounters['Ce'].count[:])
            self.resultMonitor[j,1] = self.computePopVector(currentSpikeCountB)
            self.resultMonitor[j,2] = self.computePopVector(currentSpikeCountC)
            difference = np.abs((self.resultMonitor[j,0]**2)%1. - self.resultMonitor[j,2])
            if difference > 0.5:
                difference = 1-difference
            print 'Pop. activity: ', self.resultMonitor[j,2], ', Desired activity: ', (self.resultMonitor[j,0]**2)%1., ', Difference: ', difference
            
                
                    
            if not self.testMode:
                if self.numExamples <= 1000:
                    if j%100 == 0:
                        self.saveConnections(str(j))
                else:
                    if j%1000 == 0:
                        self.saveConnections(str(j))
            
        end = time.time()
        print 'time needed to simulate:', end - start
        
        
        #------------------------------------------------------------------------------ 
        # save results
        #------------------------------------------------------------------------------ 
        print 'save results'
        
        if self.testMode:
            np.savetxt(self.dataPath + 'activity/resultPopVecs' + str(self.numExamples) + '.txt', self.resultMonitor)
        else:
            self.saveConnections(str(j))
            self.normalizeWeights()
            self.saveConnections()

    def plotResults(self):
        #------------------------------------------------------------------------------ 
        # plot results
        #------------------------------------------------------------------------------ 
        if self.rateMonitors:
            b.figure()
            for i, name in enumerate(self.rateMonitors):
                b.subplot(len(self.rateMonitors), 1, i)
                b.plot(self.rateMonitors[name].times/b.second, self.rateMonitors[name].rate, '.')
                b.title('rates of population ' + name)
            
        if self.spikeMonitors:
            b.figure()
            for i, name in enumerate(self.spikeMonitors):
                b.subplot(len(self.spikeMonitors), 1, i)
                b.raster_plot(self.spikeMonitors[name])
                b.title('spikes of population ' + name)
                if name=='Ce':
                    timePoints = np.linspace(0+(self.singleExampleTime+self.restingTime)/(2*b.second)*1000, 
                                             self.runtime/b.second*1000-(self.singleExampleTime+self.restingTime)/(2*b.second)*1000, 
                                             self.numExamples)
                    b.plot(timePoints, self.resultMonitor[:,0]*nE, 'g')
                    b.plot(timePoints, self.resultMonitor[:,1]*nE, 'r')
        
        if self.stateMonitors:
            b.figure()
            for i, name in enumerate(self.stateMonitors):
                b.plot(self.stateMonitors[name].times/b.second, self.stateMonitors[name]['v'][0], label = name + ' v 0')
                b.legend()
                b.title('membrane voltages of population ' + name)
            
        
            b.figure()
            for i, name in enumerate(self.stateMonitors):
                b.plot(self.stateMonitors[name].times/b.second, self.stateMonitors[name]['ge'][0], label = name + ' v 0')
                b.legend()
                b.title('conductances of population ' + name)
        
        plotWeights = [
        #                 'XeAe', 
        #                 'XeAi', 
        #                 'AeAe', 
        #                 'AeAi', 
        #                 'AiAe', 
        #                 'AiAi', 
        #                'BeBe', 
        #                'BeBi', 
        #                'BiBe', 
        #                'BiBi', 
        #                'CeCe', 
        #                'CeCi', 
                        'CiCe', 
        #                'CiCi', 
        #                'HeHe', 
        #                'HeHi', 
        #                'HiHe', 
        #                'HiHi', 
                        'AeHe',
        #                 'BeHe',
        #                 'CeHe',
                        'HeAe',
        #                 'HeBe',
        #                 'HeCe',
                       ]
        
        for name in plotWeights:
            b.figure()
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('own2',['#f4f4f4', '#000000'])
            my_cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('own2',['#000000', '#f4f4f4'])
            if name[1]=='e':
                nSrc = self.nE
            else:
                nSrc = self.nI
            if name[3]=='e':
                nTgt = self.nE
            else:
                nTgt = self.nI
                
            w_post = np.zeros((nSrc, nTgt))
            connMatrix = self.connections[name][:]
            for i in xrange(nSrc):
                w_post[i, connMatrix.rowj[i]] = connMatrix.rowdata[i]
            im2 = b.imshow(w_post, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
            cbar2 = b.colorbar(im2)
            b.title('weights of connection' + name)
            
            
        if self.plotError:
            error = np.abs(resultMonitor[:,1] - resultMonitor[:,0])
            correctionIdxs = np.where(error > 0.5)[0]
            correctedError = [1 - error[i] if (i in correctionIdxs) else error[i] for i in xrange(len(error))]
            correctedErrorSum = np.average(correctedError)
                 
            figure()
            scatter(resultMonitor[:,1], resultMonitor[:,0], c=range(len(error)), cmap=cm.gray)
            title('Error: ' + str(correctedErrorSum))
            xlabel('Desired activity')
            ylabel('Population activity')
             
            figure()
            error = np.abs(resultMonitor[:,1] - resultMonitor[:,0])
            correctionIdxs = np.where(error > 0.5)[0]
            correctedError = [1 - error[i] if (i in correctionIdxs) else error[i] for i in xrange(len(error))]
            correctedErrorSum = np.average(correctedError)
            scatter(resultMonitor[:,1], resultMonitor[:,0], c=resultMonitor[:,2], cmap=cm.gray)
            title('Error: ' + str(correctedErrorSum))
            xlabel('Desired activity')
            ylabel('Population activity')
        
        b.ioff()
        b.show()


    def sortWeights(self):
        #------------------------------------------------------------------------------ 
        # sort weights
        #------------------------------------------------------------------------------ 
        import threeWayEqualitySortWeights
        ending = ''
        srcWeightPath = self.dataPath + 'weights/'
        tgtWeightPath = self.dataPath + 'sortedWeights/'
        threeWayEqualitySortWeights.saveSorted(self.nE, self.nI, srcWeightPath, tgtWeightPath, 
                   ending, self.inputConnectionNames, self.interPopConnectionNames)



if __name__ == "__main__":
    relNet = RelationalNetwork()
    relNet.createNetwork()
    relNet.run()
#     net.plotResults()
    relNet.sortWeights()
    import WeightReadout
    WeightReadout.plotWeights(relNet.dataPath + 'sortedWeights/')



