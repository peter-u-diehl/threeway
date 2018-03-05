import brian as b
from brian import *
import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import scipy 
import brian.experimental.realtime_monitor as rltmMon


dataPath = ''
ending = '1000'
start_time = 0
end_time = int(ending)

resultMonitor = np.loadtxt(dataPath + 'resultPopVecs' + ending + '.txt')
    
error = np.abs(resultMonitor[start_time:end_time,0] - resultMonitor[start_time:end_time,1])
correctionIdxs = np.where(error > 0.5)[0]
correctedError = [1 - error[i] if (i in correctionIdxs) else error[i] for i in xrange(len(error))]
correctedErrorSum = np.average(correctedError)


diffLastActivity = np.abs(resultMonitor[start_time+1:end_time,1] - resultMonitor[start_time:end_time-1,1])
correctionDiffIdxs = np.where(diffLastActivity > 0.5)[0]
correctedDesDiff = [1 - diffLastActivity[i] if (i in correctionDiffIdxs) else diffLastActivity[i] for i in xrange(len(diffLastActivity))]


diffLastActivity = np.abs(resultMonitor[start_time+1:end_time,0] - resultMonitor[start_time:end_time-1,0])
correctionDiffIdxs = np.where(diffLastActivity > 0.5)[0]
correctedPopDiff = [1 - diffLastActivity[i] if (i in correctionDiffIdxs) else diffLastActivity[i] for i in xrange(len(diffLastActivity))]

print correctedErrorSum


rcParams['lines.color'] = 'w'
rcParams['text.color'] = 'w'
rcParams['xtick.color'] = 'w'
rcParams['ytick.color'] = 'w'
rcParams['axes.labelcolor'] = 'w'
rcParams['axes.edgecolor'] = 'w'

b.figure()
b.scatter(resultMonitor[start_time:end_time,2], correctedError, c=range(len(error)), cmap=cmap.gray)
b.title('Error: ' + str(correctedErrorSum))
b.xlabel('Activity in B')
b.ylabel('Error')


fi = b.figure(figsize = (5.0,4.6))
ax = plt.subplot(1,1,1)
matplotlib.rcParams.update({'font.size': 22})
b.scatter(resultMonitor[start_time:end_time,1]*1600, resultMonitor[start_time:end_time,0]*1600, c='k', cmap=cmap.gray) #range(len(error))
# b.title('Error: ' + str(correctedErrorSum))
# b.xlabel('Desired activity')
# b.ylabel('Population activity')
ax.set_xticks([0,800,1600])     
ax.set_xticklabels(['0', '800', '1600'])
ax.set_yticks([0,800,1600])     
ax.set_yticklabels(['0', '800', '1600'], va='center')
b.xlim(xmin = 0, xmax = 1600)
b.ylim(ymin = 0, ymax = 1600)
b.savefig('evaluation' + ending, dpi = 300, transparent=True)


b.figure()
b.scatter(correctedDesDiff, correctedError[1:], c=resultMonitor[start_time+1:end_time,2], cmap=cmap.gray)
b.title('Error: ' + str(correctedErrorSum))
b.xlabel('Difference from last desired activity')
b.ylabel('Error')


b.figure()
b.scatter(correctedPopDiff, correctedError[1:], c=resultMonitor[start_time+1:end_time,2], cmap=cmap.gray)
b.title('Error: ' + str(correctedErrorSum))
b.xlabel('Difference from last population activity')
b.ylabel('Error')


b.figure()
b.scatter(resultMonitor[start_time:end_time,1], resultMonitor[start_time:end_time,0], c=resultMonitor[start_time:end_time,2], cmap=cmap.gray)
b.plot(np.linspace(0, 1, end_time - start_time), np.linspace(0, 1, end_time - start_time))
b.title('Error: ' + str(correctedErrorSum))
b.xlabel('Desired activity')
b.ylabel('Population activity')
# b.savefig('evaluation' + ending)




b.show()
