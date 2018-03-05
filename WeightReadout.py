import matplotlib
matplotlib.use('Agg')

import numpy as np
from pylab import *
import matplotlib.cm as cm

def plotWeights(dataPath = './', ending = ''):
    
    
    readoutnames = []
    readoutnames.append('XeAe' + ending)
    readoutnames.append('YeBe' + ending)
    readoutnames.append('ZeCe' + ending)
    
    readoutnames.append('AeAe' + ending)
    readoutnames.append('BeBe' + ending)
    readoutnames.append('CeCe' + ending)
    # readoutnames.append('HeHe' + ending)
     
    readoutnames.append('AeHe' + ending)
    readoutnames.append('BeHe' + ending)
    readoutnames.append('CeHe' + ending)
      
    readoutnames.append('HeAe' + ending)
    readoutnames.append('HeBe' + ending)
    readoutnames.append('HeCe' + ending)
    
    readoutnames.append('AiAe' + ending)
    readoutnames.append('BiBe' + ending)
    readoutnames.append('CiCe' + ending)
    readoutnames.append('HiHe' + ending)
    
    matplotlib.rcParams.update({'figure.max_num_figures': 50})
    
    
    def computePopVector(popArray):
        size = len(popArray)
        complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
        cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
        return cur_pos
    
    
    bright_grey = '#f4f4f4'    # 
    red   = '#ff0000'  # 
    green   = '#00ff00'  # 
    black   = '#000000'    # 
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('own2',[bright_grey,black])
    
    nSrc = 1600
    nTgt = 1600
    nEH = 1600
    readoutDict = {}
    
    for name in readoutnames:
        readout = np.load(dataPath + name + '.npy')
        value_arr = np.nan * np.ones((nSrc, nTgt))
        connection_parameters = readout
        for conn in connection_parameters: 
            # don't need to pass offset as arg, now we store the parent projection
            src, tgt, value = conn
            if np.isnan(value_arr[src, tgt]):
                if not((src == tgt) and (name == 'AeAe' + ending)):
                    value_arr[src, tgt] = value
            else:
                if not((src == tgt) and (name == 'AeAe' + ending)):
                    value_arr[src, tgt] += value
        if (name == 'XeAe' + ending) or  (name == 'YeBe' + ending) or (name == 'ZeCe' + ending):
            values = np.asarray(value_arr)#.transpose()
        else:
            values = np.asarray(value_arr)
            
        fi = figure(figsize=(5.0,4.6)) # (6.3,4.6)
        fig_axis = plt.subplot(1,1,1)
        im = plt.scatter(readout[:,1], readout[:,0], s=0.5, c=readout[:,2]*2, alpha=1, marker='o', cmap=cm.get_cmap('jet'), linewidths=0, vmin= 0, vmax = 1.0)
        xlim(xmin=0, xmax=1600)
        ylim(ymin=0, ymax=1600)
        matplotlib.rcParams.update({'font.size': 22})
        axis([0.0, nSrc, 0.0, nTgt])
        fig_axis.set_xticks([0., nSrc/2, nSrc])
        fig_axis.set_yticks([0., nTgt/2, nTgt])
        savefig(dataPath + str(fi.number) + '_' + ending, dpi = 300)
    
        readoutDict[name] = np.copy(readout)
    
        if name == 'XeAe' + ending:
            XA_values = np.copy(values)#.transpose()
        if name == 'YeBe' + ending:
            YB_values = np.copy(values)#.transpose()
        if name == 'ZeCe' + ending:
            ZC_values = np.copy(values)#.transpose()
        if name == 'AeAe' + ending:
            AA_values = np.copy(values)
        if name == 'BeBe' + ending:
            BB_values = np.copy(values)
        if name == 'CeCe' + ending:
            CC_values = np.copy(values)
        if name == 'AeHe' + ending:
            AH_values = np.copy(values)
        if name == 'BeHe' + ending:
            BH_values = np.copy(values)
        if name == 'CeHe' + ending:
            CH_values = np.copy(values)
        if name == 'HeAe' + ending:
            HA_values = np.copy(values)
        if name == 'HeBe' + ending:
            HB_values = np.copy(values)
        if name == 'HeCe' + ending:
            HC_values = np.copy(values)
    
    XA_sum = np.nansum(XA_values[0:nSrc,0:nTgt], axis = 0)/nTgt
    YB_sum = np.nansum(YB_values[0:nSrc,0:nTgt], axis = 0)/nTgt
    ZC_sum = np.nansum(ZC_values[0:nSrc,0:nTgt], axis = 0)/nTgt
    AA_sum = np.nansum(AA_values[0:nTgt,0:nTgt], axis = 0)/nTgt
    BB_sum = np.nansum(BB_values[0:nTgt,0:nTgt], axis = 0)/nTgt
    CC_sum = np.nansum(CC_values[0:nTgt,0:nTgt], axis = 0)/nTgt
    AH_sum = np.nansum(AH_values[0:nTgt,0:nEH], axis = 1)/nTgt
    BH_sum = np.nansum(BH_values[0:nTgt,0:nEH], axis = 1)/nTgt
    CH_sum = np.nansum(CH_values[0:nTgt,0:nEH], axis = 1)/nTgt
    HA_sum = np.nansum(HA_values[0:nEH,0:nTgt], axis = 0)/nTgt
    HB_sum = np.nansum(HB_values[0:nEH,0:nTgt], axis = 0)/nTgt
    HC_sum = np.nansum(HC_values[0:nEH,0:nTgt], axis = 0)/nTgt
    
    AH_sum_H = np.nansum(AH_values[:nTgt,:nEH], axis = 0)/nTgt
    HA_sum_H = np.nansum(HA_values[:nEH,:nTgt], axis = 1)/nTgt
    
    fi = figure()
    plot(XA_sum, AA_sum, 'w.')
    for label, x, y in zip(range(200), XA_sum, AA_sum):
        plt.annotate(label, 
                    xy = (x, y), xytext = (-0, 0),
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    color = 'k')
    xlabel('summed input from X to A for A neurons')
    ylabel('summed input from A to A for A neurons')
    savefig(dataPath + str(fi.number))
    
    
    fi = figure()
    plot(XA_sum, AH_sum, 'w.')
    for label, x, y in zip(range(200), XA_sum, AH_sum):
        plt.annotate(label, 
                    xy = (x, y), xytext = (-0, 0),
                    textcoords = 'offset points', ha = 'right', va = 'bottom')
    xlabel('summed input from X to A for A neurons')
    ylabel('summed input from A to H for A neurons')
    savefig(dataPath + str(fi.number))
    
    fi = figure()
    plot(XA_sum, HA_sum, 'w.')
    for label, x, y in zip(range(200), XA_sum, HA_sum):
        plt.annotate(label, 
                    xy = (x, y), xytext = (-0, 0),
                    textcoords = 'offset points', ha = 'right', va = 'bottom')
    xlabel('summed input from X to A for A neurons')
    ylabel('summed input from H to A for A neurons')
    savefig(dataPath + str(fi.number))
    
    fi = figure()
    plot(AH_sum_H, HA_sum_H, 'w.')
    for label, x, y in zip(range(200), AH_sum_H, HA_sum_H):
        plt.annotate(label, 
                    xy = (x, y), xytext = (-0, 0),
                    textcoords = 'offset points', ha = 'right', va = 'bottom')
    xlabel('summed input from A to H for H neurons')
    ylabel('summed input from H to A for H neurons')
    savefig(dataPath + str(fi.number))
    
    fi = figure()
    hist(AH_sum_H)
    xlabel('Sum of the weights from A to H')
    ylabel('Number of Neurons in H')
    savefig(dataPath + str(fi.number))
    
    
    
    numPlots = 3
    plotWidth = 900
    plotHeight = 620
    fi = figure(figsize=( (plotWidth-156)*numPlots/100., 3*plotHeight/100.))
    gs = GridSpec(3*plotHeight, plotWidth*numPlots)
    im = [0,0,0]
    for subplotNum, (values,name,readout) in enumerate([(AH_values,'A_E --> H_E',readoutDict['AeHe'+ending]), (BH_values,'B_E --> H_E',readoutDict['BeHe'+ending]), (CH_values,'C_E --> H_E',readoutDict['CeHe'+ending])]):
        ax = plt.subplot(gs[plotHeight*2+100 : plotHeight*2+500, plotWidth*subplotNum+100:plotWidth*subplotNum+600])
        popVecs = np.zeros(nTgt)
        tempValues = np.nan_to_num(values)
        for x in xrange(nTgt):
            popVecs[x] = computePopVector(tempValues[:nEH,x])#.transpose())
        argSortPopVecs = np.argsort(popVecs, axis = 0)
        tempValues = np.asarray([values[:,i] for i in argSortPopVecs]).transpose()
    
        im[subplotNum] = plt.scatter(readout[:,1], readout[:,0], s=0.5, c=readout[:,2]*2, alpha=1, marker='o', cmap=cm.get_cmap('jet'), linewidths=0, vmin= 0, vmax = 1.0)
       
        xlabel('Target neuron number in ' + name[-3:])
        ylabel('Source neuron number in ' + name[0:3], labelpad = 4)
            
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)
    
        ax1_sum = np.nansum(values[0:nTgt,0:nEH], axis = 0)/nTgt
        ax0_sum = np.nansum(values[0:nTgt,0:nEH], axis = 1)/nTgt
        ax1 = plt.subplot(gs[plotHeight*2+20:plotHeight*2+100, plotWidth*subplotNum+100:plotWidth*subplotNum+600])
        ax1.plot(ax1_sum, 'k')
        ax1.set_title(name, y=1.28)
        ax1.xaxis.tick_top()
        axis([0.0, nTgt, 0.0, 0.04])
        ax1.set_yticks([0., 0.02, 0.04])
        ax1.get_xaxis().set_tick_params(pad=-1)
        
        ax0 = plt.subplot(gs[plotHeight*2+100:plotHeight*2+500, plotWidth*subplotNum+600:plotWidth*subplotNum+700])
        ax0.plot(ax0_sum, range(len(ax0_sum)), 'k')
        ax0.yaxis.tick_right()
        axis([0.0, 0.04, 0.0, nTgt])
        setp( ax0.xaxis.get_majorticklabels(), rotation=300 )
        ax0.set_xticks([0., 0.02, 0.04])
    
    im = [0,0,0]
    for subplotNum, (values,name,readout) in enumerate([(XA_values,'X_E --> A_E',readoutDict['XeAe'+ending]), (YB_values,'Y_E --> B_E',readoutDict['YeBe'+ending]), (ZC_values,'Z_E --> C_E',readoutDict['ZeCe'+ending])]):
        ax = plt.subplot(gs[100 : 500, plotWidth*subplotNum+100:plotWidth*subplotNum+600])
        im[subplotNum] = plt.scatter(readout[:,1], readout[:,0], s=0.5, c=readout[:,2]*2, alpha=1, marker='o', cmap=cm.get_cmap('jet'), linewidths=0, vmin= 0, vmax = 1.0)
        
        xlabel('Target neuron number in ' + name[-3:])
        ylabel('Source neuron number in ' + name[0:3], labelpad = 4)
            
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)
        
    
        ax1_sum = np.nansum(values[0:nSrc,0:nTgt], axis = 0)/nSrc
        ax0_sum = np.nansum(values[0:nSrc,0:nTgt], axis = 1)/nTgt
        ax1 = plt.subplot(gs[20:100, plotWidth*subplotNum+100:plotWidth*subplotNum+600])
        ax1.plot(ax1_sum, 'k')
        ax1.set_title(name, y=1.28)
        ax1.xaxis.tick_top()
        axis([0.0, len(ax1_sum), 0.0, 0.04])
        ax1.set_yticks([0., 0.02, 0.04])
        ax1.get_xaxis().set_tick_params(pad=-1)
        
        ax0 = plt.subplot(gs[100:500, plotWidth*subplotNum+600:plotWidth*subplotNum+700])
        ax0.plot(ax0_sum, range(len(ax0_sum)), 'k')
        ax0.yaxis.tick_right()
        axis([0.0, 0.04, 0.0, len(ax0_sum)])
        setp( ax0.xaxis.get_majorticklabels(), rotation=300 )
        ax0.set_xticks([0., 0.02, 0.04])
    
    im = [0,0,0]
    for subplotNum, (values,name,readout) in enumerate([(AA_values,'A_E --> A_E',readoutDict['AeAe'+ending]), (BB_values,'B_E --> B_E',readoutDict['BeBe'+ending]), (CC_values,'C_E --> C_E',readoutDict['CeCe'+ending])]):
        ax = plt.subplot(gs[plotHeight*1+100 : plotHeight*1+500, plotWidth*subplotNum+100:plotWidth*subplotNum+600])
        im[subplotNum] = plt.scatter(readout[:,1], readout[:,0], s=0.5, c=readout[:,2]*2, alpha=1, marker='o', cmap=cm.get_cmap('jet'), linewidths=0, vmin= 0, vmax = 1.0)
        
        xlabel('Target neuron number in ' + name[-3:])
        ylabel('Source neuron number in ' + name[0:3], labelpad = 4)
            
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)
        
    
        ax1_sum = np.nansum(values[0:nTgt,0:nEH], axis = 0)/nTgt
        ax0_sum = np.nansum(values[0:nTgt,0:nEH], axis = 1)/nTgt
        ax1 = plt.subplot(gs[plotHeight*1+20:plotHeight*1+100, plotWidth*subplotNum+100:plotWidth*subplotNum+600])
        ax1.plot(ax1_sum, 'k')
        ax1.set_title(name, y=1.28)
        ax1.xaxis.tick_top()
        axis([0.0, nTgt, 0.0, 0.008])
        ax1.set_yticks([0., 0.002, 0.008])
        ax1.get_xaxis().set_tick_params(pad=-1)
        
        ax0 = plt.subplot(gs[plotHeight*1+100:plotHeight*1+500, plotWidth*subplotNum+600:plotWidth*subplotNum+700])
        ax0.plot(ax0_sum, range(len(ax0_sum)), 'k')
        ax0.yaxis.tick_right()
        axis([0.0, 0.008, 0.0, nTgt])
        setp( ax0.xaxis.get_majorticklabels(), rotation=300 )
        ax0.set_xticks([0., 0.002, 0.008])
    
    savefig(dataPath + str(fi.number), dpi=800)
    
    
    
    numPlots = 3
    plotWidth = 900
    plotHeight = 500
    fi = figure(figsize=( (plotWidth-156)*numPlots/100., plotHeight/100.))
    gs = GridSpec(plotHeight, plotWidth*numPlots)
    im = [0,0,0]
    for subplotNum, (values,name,readout) in enumerate([(HA_values,'H_E --> A_E',readoutDict['HeAe'+ending]), (HB_values,'H_E --> B_E',readoutDict['HeBe'+ending]), (HC_values,'H_E --> C_E',readoutDict['HeCe'+ending])]):
        ax = plt.subplot(gs[100 : 500, plotWidth*subplotNum+100:plotWidth*subplotNum+600])
        popVecs = np.zeros(nTgt)
        tempValues = np.nan_to_num(values)
        for x in xrange(nTgt):
            popVecs[x] = computePopVector(tempValues[x,:nEH].transpose())
        argSortPopVecs = np.argsort(popVecs, axis = 0)
        tempValues = np.asarray([values[i,:] for i in argSortPopVecs])
    
    #     im[subplotNum] = ax.imshow(tempValues[:nTgt, :nTgt], interpolation="nearest", cmap=cm.get_cmap('gist_ncar'), aspect = 'auto', origin='lower')  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
        im[subplotNum] = plt.scatter(readout[:,1], readout[:,0], s=0.5, c=readout[:,2]*2, alpha=1, marker='o', cmap=cm.get_cmap('jet'), linewidths=0, vmin= 0, vmax = 1.0)
    #     im[subplotNum] = ax.pcolor(tempValues[:nTgt, :nTgt], cmap=cm.get_cmap(my_cmap))  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
        
        xlabel('Target neuron number in ' + name[-3:])
        ylabel('Source neuron number in ' + name[0:3], labelpad = 4)
            
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)
        
        axC = plt.subplot(gs[150:450, plotWidth*subplotNum+20:plotWidth*subplotNum+40])
        cbar = fi.colorbar(im[subplotNum], ax=ax, cax=axC)
        axC.yaxis.set_ticks_position("left")
    
        ax1_sum = np.nansum(values[0:nTgt,0:nTgt], axis = 0)/nTgt
        ax0_sum = np.nansum(values[0:nTgt,0:nTgt], axis = 1)/nTgt
        ax1 = plt.subplot(gs[20:100, plotWidth*subplotNum+100:plotWidth*subplotNum+600])
        ax1.plot(ax1_sum, 'k')
        ax1.set_title(name, y=1.28)
        ax1.xaxis.tick_top()
        axis([0.0, len(ax1_sum), 0.0, 0.04])
        ax1.set_yticks([0., 0.02, 0.04])
        ax1.get_xaxis().set_tick_params(pad=-1)
        
        ax0 = plt.subplot(gs[100:500, plotWidth*subplotNum+600:plotWidth*subplotNum+700])
        ax0.plot(ax0_sum, range(len(ax0_sum)), 'k')
        ax0.yaxis.tick_right()
        axis([0.0, 0.04, 0.0, len(ax0_sum)])
        setp( ax0.xaxis.get_majorticklabels(), rotation=300 )
        ax0.set_xticks([0., 0.02, 0.04])
    
    savefig(dataPath + str(fi.number))
    
    
    
    # show()#
    
    
if __name__ == "__main__":
    plotWeights()
    
    
    
