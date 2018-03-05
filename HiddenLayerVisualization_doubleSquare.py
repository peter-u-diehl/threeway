import matplotlib
matplotlib.use('Agg')

import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


#===============================================================================
# Utility functions
#===============================================================================


def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
    """
    calculate the population vector
    the pop. vector is usually close to the maximum activity
    of a layer, but in some cases, when the afferent inputs provide
    noisy input the pop. vector differed by up to 5%
    """                    
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos


def plotWeights(dataPath = './', ending=''):
    WEIGHTS_FROM_H = False
    WEIGHTS_TO_H = True
    ANNOTATE_DOTS = False
    #===============================================================================
    # load data
    #===============================================================================
    if WEIGHTS_FROM_H:
        nSrc = 1600
        nTgt = 1600
        filenames = ['HeAe' + ending + '.npy', 
                     'HeBe' + ending + '.npy', 
                     'HeCe' + ending + '.npy']
        connsFromH = []
        for name in filenames:
            print 'loading ', name
            readout = np.load(dataPath + name)
            value_arr = np.zeros((nSrc, nTgt))
            connection_parameters = readout
            #                 print connection_parameters
            for conn in connection_parameters: 
            #                     print conn
                # don't need to pass offset as arg, now we store the parent projection
                src, tgt, value = conn
                value_arr[src, tgt] += value
            values = np.asarray(value_arr)#.transpose()
            connsFromH.append(values)
            
    if WEIGHTS_TO_H:
        nSrc = 1600
        nTgt = 1600
        filenames = ['AeHe' + ending + '.npy', 
                     'BeHe' + ending + '.npy', 
                     'CeHe' + ending + '.npy']
        
        connsToH = []
        for name in filenames:
            print 'loading ', name
            readout = np.load(dataPath + name)
            value_arr = np.zeros((nSrc, nTgt))
            connection_parameters = readout
            #                 print connection_parameters
            for conn in connection_parameters: 
            #                     print conn
                # don't need to pass offset as arg, now we store the parent projection
                src, tgt, value = conn
                value_arr[src, tgt] += value
            values = np.asarray(value_arr)#.transpose()
            connsToH.append(values)
    
    #===============================================================================
    # compute assignments for H
    #===============================================================================
    
    popVecs = np.zeros((nSrc,6))
    popMaxs = np.zeros((nSrc,6))
    # print connsH
    if WEIGHTS_FROM_H:
        print 'processing weights from H'
        for x in xrange(nSrc):
            popVecs[x,0] = computePopVector(connsFromH[0][x,0:nTgt])
            popVecs[x,1] = computePopVector(connsFromH[1][x,0:nTgt])
            popVecs[x,2] = computePopVector(connsFromH[2][x,0:nTgt])
        
    #     for x in xrange(nSrc):
    #         popMaxs[x,0] = np.argmax(connsFromH[0][x,0:nTgt])
    #         popMaxs[x,1] = np.argmax(connsFromH[1][x,0:nTgt])
    #         popMaxs[x,2] = np.argmax(connsFromH[2][x,0:nTgt])
        # popMaxs *= nTgt
    #     correctedMaxs = np.asarray([popMaxs[x,:] for x in xrange(nSrc) if (not nTgt in popMaxs[x,:])])
        # print np.shape(correctedMaxs), correctedMaxs
    #     print np.shape(popMaxs), popMaxs[0:100,0:nTgt]
    #     print np.shape(connsH[0][8,0:nTgt])
    if WEIGHTS_TO_H:
        print 'processing weights to H'
        for x in xrange(nTgt):
            popVecs[x,3] = computePopVector(connsToH[0][0:nSrc,x])
            popVecs[x,4] = computePopVector(connsToH[1][0:nSrc,x])
            popVecs[x,5] = computePopVector(connsToH[2][0:nSrc,x])
        
    #     for x in xrange(nTgt):
    #         popMaxs[x,3] = np.argmax(connsToH[0][0:nSrc,x])
    #         popMaxs[x,4] = np.argmax(connsToH[1][0:nSrc,x])
    #         popMaxs[x,5] = np.argmax(connsToH[2][0:nSrc,x])
    #     correctedMaxs = np.asarray([popMaxs[x,:] for x in xrange(nSrc) if (not nTgt in popMaxs[x,:])])
        # print np.shape(correctedMaxs), correctedMaxs
    #     print np.shape(popMaxs), popMaxs[0:100,0:nTgt]
    #     print np.shape(connsH[0][8,0:nTgt])
        
    popVecs *= nSrc
    
    # figure()
    # hist(popMaxs[:,0])
    # figure()
    # hist(popMaxs[:,1])
    # figure()
    # hist(popMaxs[:,2])
    # show()
    
    
    
    
    #===============================================================================
    # plotting
    #===============================================================================
    # popVecs = popVecs[:10,:]
    
    # create x,y
    xx, yy = np.meshgrid(np.linspace(0, nSrc, 100), np.linspace(0, nTgt, 100))
    
    # calculate corresponding z
    z = (1 * xx + 1 * yy) % nTgt
    
    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.view_init(azim = -62,elev = 3)
    # plt3d.view_init(azim = -60,elev = 30)
        
    if WEIGHTS_FROM_H:
        print 'plotting 3d weights from H'
        # plt3d.plot_surface(xx, yy, z, linewidth=0.2, rstride=5, cstride=5, alpha=0.3)
        # plt3d.plot_wireframe(xx, yy, z)
        plt3d.scatter(popVecs[:,0], popVecs[:,1], popVecs[:,2], c = 'b')
        # plt3d.scatter(popMaxs[:,0], popMaxs[:,1], popMaxs[:,2], c = 'y')
        # plt3d.scatter(correctedMaxs[:,0], correctedMaxs[:,1], correctedMaxs[:,2], c = 'g')
    
    if WEIGHTS_TO_H:
        print 'plotting 3d weights to H'
        # plt3d.plot_surface(xx, yy, z, linewidth=0.2, rstride=5, cstride=5, alpha=0.3)
        # plt3d.plot_wireframe(xx, yy, z)
    #     plt3d.view_init(azim = -45,elev = 1)
        plt3d.scatter(popVecs[:,3], popVecs[:,4], popVecs[:,5], c = 'r')
        # plt3d.scatter(popMaxs[:,0], popMaxs[:,1], popMaxs[:,2], c = 'y')
        # plt3d.scatter(correctedMaxs[:,0], correctedMaxs[:,1], correctedMaxs[:,2], c = 'g')
    
    
    xlim(xmin=0, xmax=1600)
    ylim(ymin=0, ymax=1600)
    plt3d.set_zlim(bottom=0, top=1600)
    plt3d.set_xticks([0,800,1600])     
    plt3d.set_xticklabels(['0   ', '800           ', '1600               '], va='center')
    setp( plt3d.xaxis.get_majorticklabels(), rotation=45 )
    plt3d.set_yticks([0,800,1600])
    plt3d.set_yticklabels(['  0', '     800', '      1600'], va='center')
    setp( plt3d.yaxis.get_majorticklabels(), rotation=330 )
    plt3d.set_zticks([0,800,1600])
    plt3d.set_zticklabels([' 0', '     800', '      1600'], va='center')
    # setp( plt3d.zaxis.get_majorticklabels(), rotation=300 )
    #===============================================================================
    # annotate dots
    #===============================================================================
    if ANNOTATE_DOTS:
        if WEIGHTS_FROM_H:
            for label1, x1, y1, z1 in zip(range(nTgt), popVecs[:,0], popVecs[:,1], popVecs[:,2]):
                xProj1, yProj1, _ = proj3d.proj_transform(x1, y1, z1, plt3d.get_proj())   
                print x1,y1,z1, xProj1, yProj1
                plt.annotate(label1, 
                            xy = (xProj1, yProj1), xytext = (-0, 0),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            color = 'b')
        
        if WEIGHTS_TO_H:
            for label2, x2, y2, z2 in zip(range(nTgt), popVecs[:,3], popVecs[:,4], popVecs[:,5]):
                xProj2, yProj2, _ = proj3d.proj_transform(x, y, z, plt3d.get_proj())   
                print x2,y2,z2, xProj2, yProj2
                plt.annotate(label2, 
                            xy = (xProj2, yProj2), xytext = (-0, 0),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            color = 'r')
    
    # print popVecs
        
    matplotlib.rcParams.update({'font.size': 22})
    xLabel = plt3d.set_xlabel('\n     A', linespacing=3.2)
    yLabel = plt3d.set_ylabel('\n     B', linespacing=3.2)
    zLabel = plt3d.set_zlabel('\n   C', linespacing=3.2)
    
    #===============================================================================
    #  calculate the error (which is between 0 and 0.5 with 0.25 on chance)
    #===============================================================================
    
    titleString = ''
    
    if WEIGHTS_FROM_H:
        error1 = np.abs(np.mod(popVecs[:,2]/nTgt, 1.0) - np.mod((popVecs[:,0]/nTgt)**2, 1.0))
        correctionIdxs1 = np.where(error1 > 0.5)[0]
        correctedError1 = [1 - error1[i] if (i in correctionIdxs1) else error1[i] for i in xrange(nTgt)]
        errorSum1 = np.average(error1)
        correctedErrorSum1 = np.average(correctedError1)
    #     print error1
    #     print correctedError1
        print errorSum1
        print correctedErrorSum1
        titleString += ' Weights from H - error: ' + str(correctedErrorSum1)
    if WEIGHTS_TO_H:
        error2 = np.abs(np.mod(popVecs[:,5]/nTgt, 1.0) - np.mod((popVecs[:,3]/nTgt)**2, 1.0))
        correctionIdxs2 = np.where(error2 > 0.5)[0]
        correctedError2 = [1 - error2[i] if (i in correctionIdxs2) else error2[i] for i in xrange(nTgt)]
        errorSum2 = np.average(error2)
        correctedErrorSum2 = np.average(correctedError2)
    #     print error2
    #     print correctedError2
        print errorSum2
        print correctedErrorSum2
        titleString += '\n' + ' Weights to H - error: ' + str(correctedErrorSum2)
    
    plt.title(titleString)
    
    plt.savefig(dataPath + 'hiddenLayerVisualization'+ending, dpi = 300)
    temp = np.zeros((1,1))
    temp[0,0] = correctedErrorSum2
    np.savetxt(dataPath + 'hiddenLayerError'+ending+'.txt', temp)
        
#     show()

    
if __name__ == "__main__":
    plotWeights()

