import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cmap


        
def plotActivity(dataPath, ax1=None):
    averagingWindowSize = 1
    nE = 1600
    ymax = 40
#     b.rcParams['lines.color'] = 'w'
#     b.rcParams['text.color'] = 'w'
#     b.rcParams['xtick.color'] = 'w'
#     b.rcParams['ytick.color'] = 'w'
#     b.rcParams['axes.labelcolor'] = 'w'
#     b.rcParams['axes.edgecolor'] = 'w'
    plt.rcParams['font.size'] = 20
    if ax1==None:
        fig = plt.figure(figsize=(8,6.5))
        fig_axis=plt.subplot(1,1,1)
    else:
        fig_axis = ax1
        plt.sca(ax1)
    path = dataPath + 'activity/'
    popVecs = np.loadtxt(path + 'popVecs1000.txt')
    resultMonitor = np.loadtxt(path + 'resultPopVecs1000.txt')
    desiredResultFromPops = (popVecs[:,0] + popVecs[:,1])%1.*1600
    desiredResult = (resultMonitor[:,0] + resultMonitor[:,1])%1.*1600
    actualResult = resultMonitor[:,2]*1600
    
    plt.scatter(desiredResult, actualResult, c='k') #range(len(error))
    plt.scatter(1085.29537419, 994.516066213, c='deepskyblue', s=150) #range(len(error))
    plt.xlabel('Desired result')
    plt.ylabel('Population vector')
    fig_axis.set_xticks([0,nE/2.,nE])     
    fig_axis.set_xticklabels(['0', '800', '1600'])
    fig_axis.set_yticks([0,nE/2.,nE])     
    fig_axis.set_yticklabels(['0', '800', '1600'], va='center')
    plt.xlim(xmin = 0, xmax = nE)
    plt.ylim(ymin = 0, ymax = nE)
    
    
    if ax1==None:
        b.savefig(dataPath + 'testInference.png', dpi=900, transparent=False)
        
    
#     b.show()
    
if __name__ == "__main__":
    import os
    plotActivity(os.getcwd()+'/testInference/')