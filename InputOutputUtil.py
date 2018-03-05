import brian as b
import numpy as np
from brian import *
# import matplotlib
# import matplotlib.cm as cmap


def setNewInput(net,j):
    '''
        Assumes that the input net contains three input groups named X, Y, and Z which are stored in a dictionary inputGroups.
    '''
    for i,name in enumerate(net.inputPopulationNames):
        if name == 'X':
            net.popValues[j,i] = 0.5;
            rates = net.createTopoInput(net.nE, net.popValues[j,i])
        else:
            if net.testMode:
                rates = np.ones(net.nE)  * 0
            elif name == 'Y':
                net.popValues[j,i] = (net.popValues[j,0])
                rates = net.createTopoInput(net.nE, net.popValues[j,i])
            elif name == 'Z':
                net.popValues[j,i] = (net.popValues[j,0])
                rates = net.createTopoInput(net.nE, net.popValues[j,i])
        rates += net.noise
        net.inputGroups[name+'e'].rate = rates 
                
        
def plotActivity(dataPath, inputStrengths, ax1=None, fig=None):
    rateE = []
    rateI = []
    rateX = []
    xmax = 40
    ymax = 40
    for inputStrength in inputStrengths[:]:
        path = dataPath + 'in_'+str(inputStrength)+'/' +'activity/'
        spikeCountE = np.loadtxt(path + 'spikeCountAe.txt')
        spikeCountI = np.loadtxt(path + 'spikeCountAi.txt')
        spikeCountX = np.loadtxt(path + 'spikeCountXe.txt')
        print inputStrength, (sum(spikeCountE)/(len(spikeCountE))), (sum(spikeCountI)/(len(spikeCountI))), (sum(spikeCountX)/(len(spikeCountX)))
        rateE.append(sum(spikeCountE)/(len(spikeCountE)))
        rateI.append(sum(spikeCountI)/(len(spikeCountI)))
        rateX.append(sum(spikeCountX)/(len(spikeCountX)))
    b.rcParams['xtick.major.pad']='8'
        
    if ax1==None:
        unused_fig, ax1 = b.subplots(figsize=(10,8))
    else:
        b.sca(ax1)
    fig_axis = ax1
        
    b.rcParams['font.size'] = 20
#     b.rcParams['figure.autolayout'] = True
        
        
    ax1.plot(rateX, rateE, 'b', linewidth = 3., label='Excitatory')#alpha=0.5+(0.5*float(i)/float(len(lower_peaks))), 
#     ax1.plot(inputStrengths, rateX, 'r', linewidth = 3., label='Input')#alpha=0.5+(0.5*float(i)/float(len(lower_peaks))), 
    ax1.set_ylabel('Average excitatory firing rate [Hz]', color='b')
    xlabel = ax1.set_xlabel('Average / peak input rate [Hz]')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    
#     ax1.set_xticks([0., xmax/2, xmax])
#     ax1.set_yticks([0., ymax/2, ymax])
#     ax1.spines['top'].set_visible(False)
#     fig_axis.spines['right'].set_visible(False)
#     ax1.get_xaxis().tick_bottom()
#     ax1.get_yaxis().tick_left()
    
    ax2 = ax1.twinx()
    ax2.plot(rateX, rateI, 'g', linewidth = 3., label='Inhibitory')#alpha=0.5+(0.5*float(i)/float(len(lower_peaks))), 
    ax2.set_ylabel('Average inhibitory firing rate [Hz]', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')
        
        #   
    ax2.set_xticks([0., xmax/2, xmax])
    ax2.set_xticklabels([0., '20 Hz avg./ \n'+ r'$\approx$100 Hz peak', '40 Hz avg./ \n'+ r'$\approx$'+'200 Hz peak'])
    ax2.set_xlim([0., xmax])
    xlabel = ax2.set_xlabel('Input rate [Hz]')
#     ax2.set_yticks([0., ymax/2, ymax])
#     ax2.spines['top'].set_visible(False)
#     fig_axis.spines['right'].set_visible(False)
#     ax2.get_xaxis().tick_bottom()
#     ax2.get_yaxis().tick_left()
#     b.ylim(0,35)
#         b.title('spikes:' + str(sum(spikeCount)) + ', pop. value: ' + str(computePopVector(spikeCount)))
#     print ax1
    


    if ax1==None:
        unused_fig, ax1 = b.subplots(figsize=(10,8))
    else:
        b.sca(ax1)
    fig_axis = ax1


#     inset_ax1 = fig.add_axes([0.2, 0.35, .2, .1])
    inset_ax1 = fig.add_axes([0.55, 0.59, .2, .1])
    shown_range = 1
    inset_ax1.plot(rateX, rateE, 'b', linewidth = 3.)
    inset_ax1_twin = inset_ax1.twinx()
    inset_ax1_twin.plot(rateX, rateI, 'g', linewidth = 3.)

    ymax_shown = 5
    inset_ax1.set_yticks([0., ymax_shown/2., ymax_shown])
    inset_ax1.set_xlim([0., shown_range])
    inset_ax1.set_ylim([0., ymax_shown])
    inset_ax1_twin.set_ylim([0., ymax_shown])
#     inset_ax1_twin.set_xticks([0., shown_range/2., shown_range])
    inset_ax1_twin.set_xticks([0., shown_range])
    inset_ax1_twin.set_xticklabels([0., '1 Hz avg.  / \n'+ r'$\approx$5 Hz peak'], 'rotate')
    inset_ax1_twin.set_yticks([0., ymax_shown/2., ymax_shown])
#     ax1.set_yticks([0., ymax/2, ymax])

    for tl in inset_ax1.get_yticklabels():
        tl.set_color('b')
    for tl in inset_ax1_twin.get_yticklabels():
        tl.set_color('g')

#     inset_ax2 = axes([.65, .6, .2, .2])




    # b.savefig(dataPath + 'InputOutput.png', dpi=900, bbox_extra_artists=[xlabel], bbox='tight')
        
    
#     b.show()
    
if __name__ == "__main__":
    import os
    plotActivity(os.getcwd()+'/InputOutput/', np.logspace(-0.5, 2, 20, base=10))
    
    
    
